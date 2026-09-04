"""Regional LoRA math for Krea 2: subtract a LoRA's low-rank contribution outside its region.
No Forge imports.

Forge merges every LoRA into the weights. For a Linear layer the merged output is
    full(x) = base(x) + sum_r delta_r(x),   delta_r(x) = ((x @ down_r^T) @ up_r^T) * scale_r
so removing LoRA r from the tokens outside its region is
    out = full(x) - (1 - w_r[token]) * delta_r(x)
which leaves tokens with w_r = 1 untouched and gives tokens with w_r = 0 the base output.
"""

from dataclasses import dataclass

import torch


@dataclass
class LowRank:
    up: torch.Tensor  # (out, rank)
    down: torch.Tensor  # (rank, in)
    scale: float  # strength * alpha / rank

    def to(self, device, dtype) -> "LowRank":
        return LowRank(
            self.up.to(device=device, dtype=dtype),
            self.down.to(device=device, dtype=dtype),
            self.scale,
        )


def extract_lowrank(weights, strength: float):
    """weights = (up, down, alpha, mid, dora_scale, reshape) as stored by Forge's LoRAAdapter.
    Returns None for anything that is not a plain 2-D LoRA."""
    up, down, alpha, mid, dora_scale, reshape = weights
    if mid is not None or dora_scale is not None or reshape is not None:
        return None
    if up.dim() != 2 or down.dim() != 2:
        return None
    rank = down.shape[0]
    a = (float(alpha) / rank) if alpha is not None else 1.0
    return LowRank(up, down, float(strength) * a)


def lowrank_delta(x: torch.Tensor, lr: LowRank) -> torch.Tensor:
    return ((x @ lr.down.t()) @ lr.up.t()) * lr.scale


def token_weights(seg_lens, seg_indices, is_global, masks: torch.Tensor) -> torch.Tensor:
    """Per-token weight (L,) for a LoRA that lives on the given lines:
    1 on those lines' text tokens, the line's mask on image tokens, 0 elsewhere.
    A LoRA on a global line gets 1 everywhere."""
    t = sum(seg_lens)
    n = masks.shape[1]
    w = torch.zeros(t + n, dtype=torch.float32, device=masks.device)
    offsets = [0]
    for length in seg_lens:
        offsets.append(offsets[-1] + length)
    for s in seg_indices:
        if is_global[s]:
            w[:] = 1.0
            continue
        w[offsets[s] : offsets[s + 1]] = 1.0
        w[t:] = torch.maximum(w[t:], masks[s].to(w.dtype).clamp(max=1.0))
    return w


def subtract_outside(full_out: torch.Tensor, x: torch.Tensor, pairs) -> torch.Tensor:
    """In place: full_out -= sum_r (1 - w_r) * delta_r(x).
    pairs: [(LowRank, weight)] where weight is a scalar (whole tensor) or a (L,) tensor
    matching x.shape[-2]. The (1 - w) factor and the scale are folded into the small
    (..., rank) intermediate and the result is accumulated with addmm_, so no
    full-size (..., out) temporary is ever allocated (the model runs at the VRAM limit)."""
    out_dim = full_out.shape[-1]
    flat_out = full_out.view(-1, out_dim)
    for lr, w in pairs:
        h = x @ lr.down.t()  # (..., rank)
        if isinstance(w, (int, float)):
            keep = (1.0 - float(w)) * lr.scale
            if keep <= 0.0:
                continue
            h = h * keep
        else:
            keep = ((1.0 - w) * lr.scale).to(h.dtype)
            shape = [1] * (h.dim() - 2) + [w.shape[0], 1]
            h = h * keep.view(shape)
        flat_out.addmm_(h.reshape(-1, h.shape[-1]), lr.up.t(), alpha=-1.0)
    return full_out
