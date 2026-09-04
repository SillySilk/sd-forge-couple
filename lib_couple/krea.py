"""Krea 2 engine: regional prompting for a single-stream DiT through a joint-attention bias.

Krea 2 (backend/nn/krea.py) runs text and image tokens through one attention sequence, so
the batch-splitting trick used for SD / Anima has no cross-attention to hook. Instead every
prompt line is encoded on its own, the segments are joined into one text sequence, and an
additive attention bias (lib_couple/krea_bias.py) restricts each image token to the lines
whose regions cover it. All patches are class-level attribute swaps restored by unpatch();
krea.py itself is never edited.
"""

import math
import sys
import types
from dataclasses import dataclass, field

import torch

from backend import attention as _attention
from backend import memory_management
from backend.args import dynamic_args
from backend.nn import krea as _krea
from backend.patcher.lora import load_lora, model_lora_keys_unet
from backend.state_dict import state_dict_prefix_replace
from backend.utils import load_torch_file
from lib_couple.logging import logger
from modules.shared import opts

from . import krea_bias as _bias
from . import krea_lora as _lora


@dataclass
class _LoraSpec:
    name: str
    seg_indices: list[int]
    layers: dict[str, _lora.LowRank] = field(default_factory=dict)  # module path under the DiT -> LowRank


# per-file cache of low-rank layers (CPU) for the session: filename -> (layers, skipped)
_LOWRANK_CACHE: dict[str, tuple[dict[str, _lora.LowRank], int]] = {}


class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.seg_lens: list[int] = []
        self.context: torch.Tensor | None = None  # (1, T, 12, 2560)
        self.spatial: torch.Tensor | None = None  # (N, H, W) float32, cpu
        self.is_global: list[bool] = []
        self.blend: float = 0.25
        # transient, per forward
        self.bias: torch.Tensor | None = None
        self.bias_gated: torch.Tensor | None = None
        self.n_self: int = 0
        self.block_counter: int = 0
        self.txt_split: list[int] | None = None
        self.txt_seg: int | None = None
        self.tok_w: torch.Tensor | None = None  # (n_specs, L) per forward
        # regional LoRAs
        self.lora_specs: list[_LoraSpec] = []
        self.module_entries: dict[str, list[tuple[int, _lora.LowRank]]] = {}
        self.wrapped: list[torch.nn.Module] = []
        # caches / one-shot flags
        self.bias_cache: dict = {}
        self.tok_w_cache: dict = {}
        self.warned_ref: bool = False


state = _State()
_originals: dict = {}


class AttentionCoupleKrea:

    @staticmethod
    @torch.inference_mode()
    def patch_dit(model, base_mask, width: int, height: int, kwargs: dict, loras=()):
        AttentionCoupleKrea.unpatch()
        try:
            num_conds = len(kwargs) // 2 + 1
            masks = [kwargs[f"mask_{i}"] for i in range(1, num_conds)]
            spatial = torch.stack([m.reshape(m.shape[-2], m.shape[-1]).float() for m in masks], dim=0)
            segments = [kwargs[f"cond_{i}"][0][0] for i in range(1, num_conds)]

            state.seg_lens = [int(s.shape[0]) for s in segments]
            state.context = torch.cat([s.detach() for s in segments], dim=0).unsqueeze(0)
            state.spatial = spatial.detach().cpu()
            state.is_global = _bias.find_globals(state.spatial)
            state.blend = float(getattr(opts, "fc_krea_blend", 0.25))

            dit = model.model.diffusion_model
            _originals["dit_forward"] = _krea.SingleStreamDiT.forward
            _originals["block_forward"] = _krea.SingleStreamBlock.forward
            _originals["txt_forward"] = _krea.TextFusionTransformer.forward
            _originals["attention_function"] = _krea.attention_function
            _krea.SingleStreamDiT.forward = _dit_forward
            _krea.SingleStreamBlock.forward = _block_forward
            _krea.TextFusionTransformer.forward = _txt_forward
            _krea.attention_function = _attention.attention_pytorch  # attention_flash asserts mask is None
            state.active = True

            if loras:
                specs = _build_lora_specs(list(loras), model.model)
                if specs:
                    _wrap_loras(dit, specs)
                    # Move the matrices now, before Forge plans the model load. Placing them
                    # lazily inside the forward let Forge over-fill VRAM and Windows paged.
                    device = memory_management.get_torch_device()
                    dtype = model.model.computation_dtype
                    for spec in state.lora_specs:
                        for lr in spec.layers.values():
                            moved = lr.to(device, dtype)
                            lr.up, lr.down = moved.up, moved.down

            # Tell Forge's memory planner about the two attention biases plus the float32
            # build intermediate; without this Forge fills VRAM to the brim and pages.
            grid = 8 * dit.patch
            tokens = sum(state.seg_lens) + math.ceil(height / grid) * math.ceil(width / grid)
            model.add_extra_preserved_memory_during_sampling(int(4 * tokens * tokens * 2 + 64 * 1024 * 1024))

            logger.info(
                f"Krea 2: {num_conds - 1} lines ({sum(state.is_global)} global), "
                f"blend {state.blend} -> {int(round(state.blend * len(dit.blocks)))} gated blocks"
                + (f", regional LoRAs: {len(state.lora_specs)}" if state.lora_specs else "")
            )
            return model

        except Exception:
            logger.exception("Krea 2 patch failed...")
            AttentionCoupleKrea.unpatch()
            return None

    @staticmethod
    def active_loras() -> list[tuple[str, list[int]]]:
        return [(spec.name, list(spec.seg_indices)) for spec in state.lora_specs]

    @staticmethod
    def unpatch():
        _unwrap_loras()
        if "dit_forward" in _originals:
            _krea.SingleStreamDiT.forward = _originals.pop("dit_forward")
        if "block_forward" in _originals:
            _krea.SingleStreamBlock.forward = _originals.pop("block_forward")
        if "txt_forward" in _originals:
            _krea.TextFusionTransformer.forward = _originals.pop("txt_forward")
        if "attention_function" in _originals:
            _krea.attention_function = _originals.pop("attention_function")
        state.reset()


# ---------------------------------------------------------------- helpers


def _grid(x: torch.Tensor, patch: int) -> tuple[int, int]:
    return math.ceil(x.shape[-2] / patch), math.ceil(x.shape[-1] / patch)


def _biases_for(h: int, w: int, dtype: torch.dtype, device: torch.device):
    key = (h, w, str(dtype), str(device))
    if key not in state.bias_cache:
        masks = _bias.resize_masks(state.spatial.to(device), h, w)
        plain = _bias.build_bias(state.seg_lens, masks, state.is_global, False)
        gated = _bias.build_bias(state.seg_lens, masks, state.is_global, True)
        state.bias_cache[key] = (
            _bias.pad_aligned(plain, dtype, device),
            _bias.pad_aligned(gated, dtype, device),
        )
    return state.bias_cache[key]


def _cond_rows(transformer_options: dict, rows: int) -> torch.Tensor:
    cond_mark = transformer_options.get("cond_mark", None)
    if cond_mark is None:
        return torch.ones(rows, dtype=torch.bool)
    return cond_mark.flatten()[:rows].cpu() < 0.5


# ---------------------------------------------------------------- regional LoRAs


def _lowrank_layers(filename: str, unet_model):
    if filename not in _LOWRANK_CACHE:
        sd = load_torch_file(filename, safe_load=True)
        if any(k.startswith("lora_unet__") for k in sd):
            sd = state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})
        patches = load_lora(sd, model_lora_keys_unet(unet_model))
        if isinstance(patches, tuple):  # Forge's load_lora returns (patch_dict, remaining_keys)
            patches = patches[0]
        layers: dict[str, _lora.LowRank] = {}
        skipped = 0
        for key, adapter in patches.items():
            if not (key.startswith("diffusion_model.") and key.endswith(".weight")) or not hasattr(adapter, "weights"):
                skipped += 1
                continue
            lr = _lora.extract_lowrank(adapter.weights, 1.0)
            if lr is None:
                skipped += 1
                continue
            layers[key[len("diffusion_model.") : -len(".weight")]] = lr
        _LOWRANK_CACHE[filename] = (layers, skipped)
    return _LOWRANK_CACHE[filename]


def _build_lora_specs(line_loras: list[list[str]], unet_model) -> list[_LoraSpec]:
    nets = getattr(sys.modules.get("networks"), "loaded_networks", [])
    by_name = {}
    for n in nets:
        by_name[n.name] = n
        if getattr(n, "mentioned_name", None):
            by_name[n.mentioned_name] = n

    per_name: dict[str, set[int]] = {}
    for seg, names in enumerate(line_loras):
        if seg >= len(state.is_global) or state.is_global[seg]:
            continue  # global line: LoRA stays global
        for name in names:
            per_name.setdefault(name, set()).add(seg)

    specs = []
    for name, segs in per_name.items():
        net = by_name.get(name)
        if net is None:
            logger.warning(f"Krea 2: LoRA '{name}' is not loaded; staying global")
            continue
        layers, skipped = _lowrank_layers(net.network_on_disk.filename, unet_model)
        strength = float(net.unet_multiplier)
        copies = {path: _lora.LowRank(lr.up, lr.down, lr.scale * strength) for path, lr in layers.items()}
        if skipped:
            logger.warning(f"Krea 2: LoRA '{name}': {skipped} unsupported layer(s) stay global")
        if not copies:
            logger.warning(f"Krea 2: LoRA '{name}': no supported layers; staying global")
            continue
        specs.append(_LoraSpec(name, sorted(segs), copies))
    return specs


def _wrap_loras(dit, specs: list[_LoraSpec]):
    entries: dict[str, list[tuple[int, _lora.LowRank]]] = {}
    for i, spec in enumerate(specs):
        for path, lr in spec.layers.items():
            entries.setdefault(path, []).append((i, lr))

    missing = 0
    for path in list(entries):
        try:
            m = dit.get_submodule(path)
        except AttributeError:
            missing += 1
            del entries[path]
            continue
        m._fc_path = path
        m._fc_mode = "txt" if path.startswith("txtfusion.") else "seq"
        m.forward = types.MethodType(_lora_linear_forward, m)
        state.wrapped.append(m)

    state.module_entries = entries
    state.lora_specs = specs
    for spec in specs:
        logger.info(f"Krea 2: LoRA '{spec.name}' -> line(s) {[i + 1 for i in spec.seg_indices]} ({len(spec.layers)} layers)")
    if missing:
        logger.warning(f"Krea 2: {missing} LoRA layer path(s) not found in the model; those stay global")


def _unwrap_loras():
    for m in state.wrapped:
        m.__dict__.pop("forward", None)
        for attr in ("_fc_path", "_fc_mode"):
            if hasattr(m, attr):
                delattr(m, attr)
    state.wrapped = []
    state.module_entries = {}
    state.lora_specs = []


def _lr_on(lr: _lora.LowRank, x: torch.Tensor) -> _lora.LowRank:
    if lr.up.device != x.device or lr.up.dtype != x.dtype:
        moved = lr.to(x.device, x.dtype)
        lr.up, lr.down = moved.up, moved.down  # spec-local copies, safe to cast in place
    return lr


def _lora_linear_forward(self, x):
    out = type(self).forward(self, x)
    entries = state.module_entries.get(getattr(self, "_fc_path", None))
    if not entries:
        return out
    if self._fc_mode == "seq":
        tw = state.tok_w
        if tw is None or x.shape[-2] != tw.shape[-1]:
            return out
        pairs = [(_lr_on(lr, x), tw[i]) for i, lr in entries]
    else:
        if state.txt_seg is None:
            return out
        pairs = [
            (_lr_on(lr, x), 1.0 if state.txt_seg in state.lora_specs[i].seg_indices else 0.0) for i, lr in entries
        ]
    return _lora.subtract_outside(out, x, pairs)


def _tok_weights_for(h: int, w: int, device: torch.device) -> torch.Tensor:
    key = (h, w, str(device))
    if key not in state.tok_w_cache:
        masks = _bias.resize_masks(state.spatial.to(device), h, w)
        rows = [_lora.token_weights(state.seg_lens, spec.seg_indices, state.is_global, masks) for spec in state.lora_specs]
        state.tok_w_cache[key] = torch.stack(rows, dim=0)
    return state.tok_w_cache[key]


# ---------------------------------------------------------------- patched forwards


def _run_regional(self, x, timesteps, context, attention_mask, transformer_options, kwargs):
    orig = _originals["dit_forward"]
    rows = x.shape[0]
    h, w = _grid(x, self.patch)
    ctx = state.context.to(device=x.device, dtype=context.dtype).expand(rows, -1, -1, -1)
    state.bias, state.bias_gated = _biases_for(h, w, x.dtype, x.device)
    state.n_self = int(round(state.blend * len(self.blocks)))
    state.block_counter = 0
    state.txt_split = list(state.seg_lens)
    if state.lora_specs:
        state.tok_w = _tok_weights_for(h, w, x.device)
    try:
        return orig(self, x, timesteps, ctx, attention_mask, transformer_options, **kwargs)
    finally:
        state.bias = None
        state.bias_gated = None
        state.txt_split = None
        state.tok_w = None


def _dit_forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
    orig = _originals["dit_forward"]
    if not state.active or state.context is None:
        return orig(self, x, timesteps, context, attention_mask, transformer_options, **kwargs)

    if dynamic_args.ref_latents:
        if not state.warned_ref:
            state.warned_ref = True
            logger.warning("Krea 2 edit / reference mode is not supported; generating without regions")
        return orig(self, x, timesteps, context, attention_mask, transformer_options, **kwargs)

    try:
        rows = x.shape[0]
        is_cond = _cond_rows(transformer_options, rows)

        if bool(is_cond.all()):
            return _run_regional(self, x, timesteps, context, attention_mask, transformer_options, kwargs)
        if not bool(is_cond.any()):
            return orig(self, x, timesteps, context, attention_mask, transformer_options, **kwargs)

        # mixed cond / uncond rows in one batch: run the two halves separately
        idx_c = is_cond.nonzero().flatten().to(x.device)
        idx_u = (~is_cond).nonzero().flatten().to(x.device)

        def _rows(t, idx):
            return t[idx] if torch.is_tensor(t) and t.shape[0] == rows else t

        out_c = _run_regional(
            self, x[idx_c], _rows(timesteps, idx_c), context[idx_c], attention_mask, transformer_options, kwargs
        )
        out_u = orig(
            self, x[idx_u], _rows(timesteps, idx_u), context[idx_u], attention_mask, transformer_options, **kwargs
        )
        out = torch.empty((rows, *out_c.shape[1:]), dtype=out_c.dtype, device=out_c.device)
        out[idx_c] = out_c
        out[idx_u] = out_u
        return out

    except Exception:
        logger.exception("Krea 2 regional forward failed; generating without regions")
        AttentionCoupleKrea.unpatch()
        return orig(self, x, timesteps, context, attention_mask, transformer_options, **kwargs)


def _block_forward(self, x, vec, freqs, mask=None, transformer_options={}):
    if mask is None and state.bias is not None:
        i = state.block_counter
        state.block_counter += 1
        if state.bias.dtype != x.dtype:
            state.bias = state.bias.to(x.dtype)
            state.bias_gated = state.bias_gated.to(x.dtype)
        mask = state.bias_gated if i < state.n_self else state.bias
    return _originals["block_forward"](self, x, vec, freqs, mask, transformer_options=transformer_options)


def _txt_forward(self, x, mask=None, transformer_options={}):
    orig = _originals["txt_forward"]
    split = state.txt_split
    if split is None or x.shape[1] != sum(split):
        return orig(self, x, mask, transformer_options=transformer_options)
    parts = x.split(split, dim=1)
    outs = []
    try:
        for i, part in enumerate(parts):
            state.txt_seg = i
            outs.append(orig(self, part.contiguous(), None, transformer_options=transformer_options))
    finally:
        state.txt_seg = None
    return torch.cat(outs, dim=1)
