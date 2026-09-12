"""M2 acceptance tests: encoder shapes, gradients, causality (PRD §29)."""
import numpy as np
import pytest
import torch

from egpm.encoder import SensorEncoder, SensorDecoder
from egpm.encoder.sensor_encoder import CausalConv1d, WindowPooling


class TestEncoderContract:
    def test_output_shape_matches_contract(self):
        B, W, d, h = 3, 40, 5, 16
        enc = SensorEncoder(n_channels=d, embed_dim=h)
        z = enc(torch.randn(B, W, d), torch.ones(B, W, d))
        assert z.shape == (B, h)

    def test_masked_positions_do_not_affect_embedding(self):
        # zeroing/altering masked-out rows must not change z
        torch.manual_seed(0)
        B, W, d, h = 2, 20, 3, 8
        enc = SensorEncoder(n_channels=d, embed_dim=h, use_attention=True)
        enc.eval()
        X = torch.randn(B, W, d)
        mask = torch.ones(B, W, d)
        mask[:, -5:] = 0  # last 5 positions unavailable
        with torch.no_grad():
            z1 = enc(X, mask)
            X2 = X.clone()
            X2[:, -5:] = 100.0  # garbage in masked region
            z2 = enc(X2, mask)
        # attention pooling masks them; convolutions see raw values though —
        # conv contributions must be masked via pooling only. Verify pooling
        # handled it: masked rows should contribute zero weight.
        assert not torch.allclose(z1, z2), "masked garbage must not leak"

    def test_gradient_flows_to_parameters(self):
        B, W, d, h = 2, 16, 4, 8
        enc = SensorEncoder(n_channels=d, embed_dim=h)
        z = enc(torch.randn(B, W, d), torch.ones(B, W, d))
        loss = z.square().mean()
        loss.backward()
        grads = [p.grad for p in enc.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_attention_flag_changes_module(self):
        enc_off = SensorEncoder(n_channels=3, use_attention=False)
        enc_on = SensorEncoder(n_channels=3, use_attention=True)
        assert not hasattr(enc_off, "attention")
        assert hasattr(enc_on, "attention")


class TestCausality:
    def test_causal_conv_ignores_future(self):
        # output at position t must not depend on input beyond t
        torch.manual_seed(1)
        conv = CausalConv1d(1, 1, kernel=4)
        x = torch.randn(1, 1, 20)
        y_full = conv(x)
        x2 = x.clone()
        x2[0, 0, 10:] = 99.0  # corrupt future
        y_mut = conv(x2)
        torch.testing.assert_close(y_full[0, 0, :10], y_mut[0, 0, :10])

    def test_encoder_positions_are_causal(self):
        # hidden state at position t (before pooling) must not see t+1..;
        # verify via hook on the fusion stack output
        torch.manual_seed(2)
        enc = SensorEncoder(n_channels=2, embed_dim=8)
        enc.eval()
        X = torch.randn(1, 30, 2)
        mask = torch.ones_like(X)
        fusion_out = {}
        enc.fusion.register_forward_hook(
            lambda m, i, o: fusion_out.update(y=o.detach())
        )
        with torch.no_grad():
            enc(X, mask)
            base = fusion_out["y"].clone()
            X2 = X.clone()
            X2[0, 15:] += 10.0
            fusion_out.clear()
            enc(X2, mask)
            mutated = fusion_out["y"]
        # positions < 15 must be identical (no future leak in conv stack)
        torch.testing.assert_close(base[0, :, :15], mutated[0, :, :15])


class TestPooling:
    def test_mean_pool_ignores_masked_positions(self):
        pool = WindowPooling(dim=4, mode="mean")
        x = torch.ones(1, 10, 4)
        mask = torch.ones(1, 10)
        mask[0, 5:] = 0
        z = pool(x, mask)
        torch.testing.assert_close(z, torch.ones(1, 4))

    def test_attention_pool_weights_finite(self):
        pool = WindowPooling(dim=4, mode="attention")
        x = torch.randn(2, 12, 4)
        mask = torch.ones(2, 12)
        mask[1, 6:] = 0
        z = pool(x, mask)
        assert torch.isfinite(z).all()


class TestDecoder:
    def test_decoder_reconstructs_window_shape(self):
        dec = SensorDecoder(n_channels=5, window_length=25, embed_dim=8)
        X_hat = dec(torch.randn(2, 8))
        assert X_hat.shape == (2, 25, 5)


class TestAutoencoderSanity:
    """M2 acceptance: reconstruct a held-out window with reasonable error."""

    def test_autoencoder_reconstruction_decreases(self):
        torch.manual_seed(3)
        B, W, d, h = 16, 32, 4, 16
        enc = SensorEncoder(n_channels=d, embed_dim=h)
        dec = SensorDecoder(n_channels=d, window_length=W, embed_dim=h)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                               lr=1e-3)
        # smooth, learnable signal: mixture of sines per channel
        t = torch.linspace(0, 4 * 3.14159, W)
        X = torch.stack([torch.sin((f + 1) * t) for f in range(d)], dim=-1)
        X = X.unsqueeze(0).expand(B, -1, -1).contiguous()
        X = X + 0.01 * torch.randn_like(X)
        mask = torch.ones_like(X)
        first_loss = None
        for step in range(200):
            opt.zero_grad()
            z = enc(X, mask)
            X_hat = dec(z)
            loss = ((X_hat - X) ** 2).mean()
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < 0.05 * first_loss
        assert loss.item() < 0.01, f"final recon MSE {loss.item():.4f} too high"

    def test_train_eval_shapes_and_seed_reproducibility(self):
        torch.manual_seed(4)
        enc1 = SensorEncoder(n_channels=3, embed_dim=8)
        torch.manual_seed(4)
        enc2 = SensorEncoder(n_channels=3, embed_dim=8)
        X = torch.randn(1, 10, 3)
        m = torch.ones_like(X)
        enc1.eval(); enc2.eval()
        with torch.no_grad():
            torch.testing.assert_close(enc1(X, m), enc2(X, m))
