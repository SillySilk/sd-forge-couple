import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from lib_couple import krea_lora as Lo


class Extract(unittest.TestCase):
    def test_plain_lora_scale(self):
        up, down = torch.randn(8, 4), torch.randn(4, 6)
        lr = Lo.extract_lowrank((up, down, 2.0, None, None, None), 0.5)
        self.assertIsNotNone(lr)
        self.assertAlmostEqual(lr.scale, 0.5 * 2.0 / 4, places=6)

    def test_alpha_none(self):
        lr = Lo.extract_lowrank((torch.randn(8, 4), torch.randn(4, 6), None, None, None, None), 1.0)
        self.assertEqual(lr.scale, 1.0)

    def test_rejects_mid_dora_reshape_conv(self):
        up, down = torch.randn(8, 4), torch.randn(4, 6)
        self.assertIsNone(Lo.extract_lowrank((up, down, 1.0, torch.randn(1), None, None), 1.0))
        self.assertIsNone(Lo.extract_lowrank((up, down, 1.0, None, torch.randn(8), None), 1.0))
        self.assertIsNone(Lo.extract_lowrank((up, down, 1.0, None, None, [8, 6]), 1.0))
        self.assertIsNone(
            Lo.extract_lowrank((torch.randn(8, 4, 1, 1), torch.randn(4, 6, 3, 3), 1.0, None, None, None), 1.0)
        )


class Delta(unittest.TestCase):
    def test_matches_dense(self):
        torch.manual_seed(0)
        up, down = torch.randn(8, 4), torch.randn(4, 6)
        lr = Lo.LowRank(up, down, 0.3)
        x = torch.randn(2, 5, 6)
        dense = x @ ((up @ down) * 0.3).t()
        self.assertTrue(torch.allclose(Lo.lowrank_delta(x, lr), dense, atol=1e-5))


class SubtractOutside(unittest.TestCase):
    def test_sequence_weights(self):
        torch.manual_seed(0)
        W = torch.randn(8, 6)
        up, down = torch.randn(8, 4), torch.randn(4, 6)
        lr = Lo.LowRank(up, down, 0.3)
        x = torch.randn(1, 5, 6)
        base = x @ W.t()
        full = x @ (W + 0.3 * up @ down).t()
        w = torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0])
        out = Lo.subtract_outside(full.clone(), x, [(lr, w)])
        self.assertTrue(torch.allclose(out[0, 0], full[0, 0], atol=1e-5))  # inside: unchanged
        self.assertTrue(torch.allclose(out[0, 1], base[0, 1], atol=1e-5))  # outside: LoRA removed
        self.assertTrue(torch.allclose(out[0, 2], 0.5 * (full[0, 2] + base[0, 2]), atol=1e-5))  # soft edge

    def test_scalar_weight_and_two_loras(self):
        torch.manual_seed(1)
        W = torch.randn(8, 6)
        a = Lo.LowRank(torch.randn(8, 4), torch.randn(4, 6), 0.2)
        b = Lo.LowRank(torch.randn(8, 3), torch.randn(3, 6), 0.7)
        x = torch.randn(3, 6)
        full = x @ (W + 0.2 * a.up @ a.down + 0.7 * b.up @ b.down).t()
        only_a = x @ (W + 0.2 * a.up @ a.down).t()
        out = Lo.subtract_outside(full.clone(), x, [(a, 1.0), (b, 0.0)])
        self.assertTrue(torch.allclose(out, only_a, atol=1e-5))

    def test_in_place_and_3d_scalar(self):
        torch.manual_seed(2)
        W = torch.randn(8, 6)
        a = Lo.LowRank(torch.randn(8, 4), torch.randn(4, 6), 0.5)
        x = torch.randn(2, 3, 6)
        full = x @ (W + 0.5 * a.up @ a.down).t()
        base = x @ W.t()
        buf = full.clone()
        out = Lo.subtract_outside(buf, x, [(a, 0.0)])
        self.assertIs(out, buf)
        self.assertTrue(torch.allclose(out, base, atol=1e-5))


class TokenWeights(unittest.TestCase):
    def test_layout(self):
        masks = torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]])  # (N=3, 4)
        seg_lens, is_global = [3, 2, 2], [True, False, False]
        w = Lo.token_weights(seg_lens, [2], is_global, masks)  # LoRA on line 3
        self.assertEqual(tuple(w.shape), (11,))
        self.assertTrue(torch.equal(w[:3], torch.zeros(3)))  # global text: 0
        self.assertTrue(torch.equal(w[3:5], torch.zeros(2)))  # other line's text: 0
        self.assertTrue(torch.equal(w[5:7], torch.ones(2)))  # own text: 1
        self.assertTrue(torch.equal(w[7:], torch.tensor([0.0, 0.0, 1.0, 0.5])))

    def test_union_of_lines(self):
        masks = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
        w = Lo.token_weights([1, 1, 1], [1, 2], [True, False, False], masks)
        self.assertTrue(torch.equal(w, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])))

    def test_global_line_lora_is_global(self):
        masks = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
        w = Lo.token_weights([1, 1, 1], [0], [True, False, False], masks)
        self.assertTrue(torch.equal(w, torch.ones(5)))


if __name__ == "__main__":
    unittest.main()
