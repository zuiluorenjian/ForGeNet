import unittest

import torch
from torch.nn import functional as F

from loss import SupConLoss
from networks.gradient_gate import LearnableGradientGate, combine_objectives, measure_gradient_norms
from networks.pretrain_model import QVLoRAParametrization


class Stage1Tests(unittest.TestCase):
    def test_lgr_positive_pairs_and_batch_reduction(self):
        criterion = SupConLoss(min_class=1, ratio_supervised_majority=0.0)
        labels = torch.tensor([0, 0, 1, 1])
        mask, _ = criterion._create_numerator_mask(labels[:, None], 0.0, 1, 0)
        self.assertEqual(mask[0, 1].item(), 0)
        self.assertEqual(mask[2, 3].item(), 1)
        self.assertTrue(torch.equal(mask.diag(), torch.ones(4)))
        features = F.normalize(torch.randn(4, 2, 8), dim=-1).requires_grad_()
        loss, _, _, parts = criterion(features, labels, return_components=True)
        torch.testing.assert_close(parts["real"] + parts["fake"], loss)
        torch.testing.assert_close(combine_objectives(parts["real"], parts["fake"], loss.new_tensor(0.5)), loss)

    def test_gate_feedback_does_not_train_encoder_or_leak_into_detector_loss(self):
        shared = torch.tensor([0.3, 0.7], requires_grad=True)
        real_loss, fake_loss = shared.square().sum(), (shared - 2).square().sum()
        norms, _ = measure_gradient_norms(real_loss, fake_loss, [shared])
        self.assertIsNone(shared.grad)
        features = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        gate = LearnableGradientGate(8)
        gate.observe(norms)
        weight = gate(features, labels)
        self.assertAlmostEqual(weight.item(), 0.5, places=6)
        gate.balance_loss(weight).backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(shared.grad)
        self.assertGreater(gate.net[-1].weight.grad.abs().sum().item(), 0)
        gate.zero_grad(set_to_none=True)
        combine_objectives(real_loss, fake_loss, weight).backward()
        self.assertGreater(shared.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in gate.parameters()))

    def test_single_class_components_are_finite(self):
        criterion = SupConLoss(min_class=1, ratio_supervised_majority=0.0)
        for value in (0, 1):
            features = F.normalize(torch.randn(4, 2, 8), dim=-1).requires_grad_()
            loss, _, _, parts = criterion(features, torch.full((4,), value), return_components=True)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(parts["fake_count" if value == 0 else "real_count"], 0)
            loss.backward()
            self.assertTrue(torch.isfinite(features.grad).all())

    def test_qv_lora_preserves_k_and_merges_exactly(self):
        adapter = QVLoRAParametrization(embed_dim=8, rank=2, alpha=4)
        base = torch.randn(24, 8)
        torch.testing.assert_close(adapter(base), base)
        with torch.no_grad():
            adapter.q_B.normal_()
            adapter.v_B.normal_()
        merged = adapter(base)
        torch.testing.assert_close(merged[8:16], base[8:16])
        torch.testing.assert_close(merged[:8], base[:8] + 2 * (adapter.q_B @ adapter.q_A))
        torch.testing.assert_close(merged[16:], base[16:] + 2 * (adapter.v_B @ adapter.v_A))


if __name__ == "__main__":
    unittest.main()
