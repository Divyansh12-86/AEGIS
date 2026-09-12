"""M3 acceptance tests: VQ straight-through correctness, EMA, collapse monitor (PRD §29).

The straight-through tests are the crux (PRD §29 "VQ tests"):
  * forward value of z_hat is exactly c_v (the codebook vector);
  * gradient applied to z_hat reaches z unchanged (identity Jacobian).
"""
import numpy as np
import pytest
import torch

from egpm.quantization import VQCodebook


class TestStraightThrough:
    def test_forward_value_is_codebook_vector(self):
        torch.manual_seed(0)
        vq = VQCodebook(embed_dim=4, n_codes=8)
        z = torch.randn(6, 4)
        cb_before = vq.codebook.clone()  # capture before (train mode runs EMA)
        out = vq(z)
        expected = cb_before[out.event_ids]
        torch.testing.assert_close(out.straight_through, expected)

    def test_gradient_flows_to_z_unchanged(self):
        torch.manual_seed(1)
        vq = VQCodebook(embed_dim=4, n_codes=8)
        z = torch.randn(5, 4, requires_grad=True)
        out = vq(z)
        # apply loss directly to z_hat; grad on z must be the identity
        grad_target = torch.randn_like(out.straight_through)
        out.straight_through.backward(grad_target)
        torch.testing.assert_close(z.grad, grad_target)

    def test_quantized_detached_from_encoder(self):
        # c_v itself carries no gradient (buffer-indexed), so it can never
        # bypass the ST path into the encoder
        torch.manual_seed(2)
        vq = VQCodebook(embed_dim=3, n_codes=4)
        z = torch.randn(2, 3, requires_grad=True)
        out = vq(z)
        assert not out.quantized.requires_grad
        # the only gradient path to z is the ST identity
        out.straight_through.sum().backward()
        assert z.grad is not None
        torch.testing.assert_close(z.grad, torch.ones_like(z))

    def test_event_ids_are_argmin(self):
        torch.manual_seed(3)
        vq = VQCodebook(embed_dim=5, n_codes=16)
        vq.eval()  # deterministic, no EMA side effect
        z = torch.randn(32, 5)
        ids = vq(z).event_ids
        d2 = torch.cdist(z, vq.codebook) ** 2
        torch.testing.assert_close(ids, d2.argmin(dim=1))


class TestEMAUpdate:
    def test_codebook_moves_toward_assigned_mean(self):
        torch.manual_seed(4)
        vq = VQCodebook(embed_dim=4, n_codes=4, ema_decay=0.0)  # decay 0: hard assign
        before = vq.codebook.clone()
        z = torch.randn(20, 4)
        ids = vq.event_id(z)
        vq.train()
        vq(z)  # triggers EMA update with decay=0 -> codebook = mean of assigned
        for k in range(4):
            assigned = z[ids == k]
            if len(assigned) > 0:
                expected = assigned.mean(dim=0)
                torch.testing.assert_close(vq.codebook[k], expected, rtol=1e-4, atol=1e-5)

    def test_unused_codes_survive_laplace_smoothing(self):
        vq = VQCodebook(embed_dim=2, n_codes=64, ema_decay=0.5)
        z = torch.randn(4, 2)  # at most 4 codes used
        vq.train()
        vq(z)
        assert torch.isfinite(vq.codebook).all()

    def test_eval_mode_skips_ema(self):
        vq = VQCodebook(embed_dim=3, n_codes=8)
        vq.eval()
        before = vq.codebook.clone()
        z = torch.randn(10, 3)
        vq(z)
        torch.testing.assert_close(vq.codebook, before)


class TestCollapseMonitor:
    def test_perplexity_one_when_all_collapse(self):
        vq = VQCodebook(embed_dim=2, n_codes=10)
        z = torch.zeros(50, 2)  # all identical -> one code used
        stats = vq(z).codebook_stats
        assert stats["perplexity"] == pytest.approx(1.0, rel=1e-6)
        assert stats["effective_size"] == 1

    def test_perplexity_matches_usage_entropy(self):
        vq = VQCodebook(embed_dim=2, n_codes=4)
        # craft z so exactly 2 codes are used equally
        z = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
        stats = vq(z).codebook_stats
        assert stats["perplexity"] == pytest.approx(2.0, rel=1e-4)
        assert stats["effective_size"] == 2

    def test_usage_histogram_is_distribution(self):
        vq = VQCodebook(embed_dim=3, n_codes=7)
        z = torch.randn(23, 3)
        hist = vq(z).codebook_stats["usage_histogram"]
        assert hist.sum() == pytest.approx(1.0, abs=1e-9)
        assert len(hist) == 7

    def test_perplexity_stable_no_collapse_over_training(self):
        """M3 acceptance criterion: perplexity stable, no collapse, over a
        full (small) training run with commitment loss + EMA."""
        torch.manual_seed(5)
        K, h = 16, 8
        vq = VQCodebook(embed_dim=h, n_codes=K, ema_decay=0.9)
        # 4 well-separated clusters; codebook initialized with codes near each
        centers = torch.tensor(
            [[i * 10.0] * h for i in range(4)] + [[0.0] * h] * (K - 4)
        )
        vq.codebook.copy_(centers + 0.01 * torch.randn(K, h))
        perplexities = []
        for _ in range(60):
            idx = torch.randint(0, 4, (64,))
            z = centers[idx] + 0.1 * torch.randn(64, h)
            vq.train()
            out = vq(z)
            perplexities.append(out.codebook_stats["perplexity"])
        vq.eval()
        # no collapse: >= 3 effective codes at the end, perplexity > 3
        final_stats = vq(z).codebook_stats
        assert final_stats["effective_size"] >= 3
        assert perplexities[-1] > 3.0
        # stable: last-10 perplexities don't crash toward 1
        assert min(perplexities[-10:]) > 2.0

    def test_loss_terms_shapes_and_finite(self):
        vq = VQCodebook(embed_dim=4, n_codes=6)
        z = torch.randn(9, 4, requires_grad=True)
        out = vq(z)
        terms = vq.loss_terms(z, out)
        for name in ("codebook", "commitment"):
            assert torch.isfinite(terms[name]).all(), name
        (terms["commitment"] + terms["codebook"]).backward()
        assert z.grad is not None  # commitment reaches z (beta term)


class TestContract:
    def test_output_shapes(self):
        B, h, K = 7, 5, 12
        vq = VQCodebook(embed_dim=h, n_codes=K)
        out = vq(torch.randn(B, h))
        assert out.event_ids.shape == (B,)
        assert out.straight_through.shape == (B, h)
        assert out.event_ids.min() >= 0 and out.event_ids.max() < K

    def test_invalid_input_rejected(self):
        vq = VQCodebook(embed_dim=4, n_codes=8)
        with pytest.raises(ValueError):
            vq(torch.randn(3, 5))  # wrong dim

    def test_no_nan_under_extreme_values(self):
        vq = VQCodebook(embed_dim=3, n_codes=8)
        z = torch.tensor([[1e8, -1e8, 0.0], [1e-20, 1e-20, 1e-20]])
        out = vq(z)
        assert torch.isfinite(out.straight_through).all()
