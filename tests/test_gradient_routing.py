import torch
import pytest
from egpm.training.trainer import Trainer, TrainerConfig


@pytest.mark.parametrize("state_dim,prog_dim", [(12, 4), (8, 8)])
def test_gradient_routing(state_dim, prog_dim):
    cfg = TrainerConfig(
        window_length=4,
        embed_dim=16,
        hidden=32,
        n_conv_blocks=2,
        n_codes=16,
        aux_rul_lambda=0.1,
        state_dim=state_dim,
        stage1_steps=1,
        batch_size=4,
        seed=42,
    )
    B, W, d = 4, 4, 6
    trainer = Trainer(cfg, n_channels=d)

    X = torch.randn(B, W, d, requires_grad=False)
    mask = torch.ones_like(X)
    y_rul = torch.tensor([50.0, 40.0, 30.0, 20.0])

    # 1. Forward encoder to continuous latent z
    # Retain grad on z to measure gradient routing directly into each subspace
    z = trainer.encoder(X, mask)
    z.retain_grad()

    z_state = z[:, :state_dim]
    z_prog = z[:, state_dim:]

    # -------------------------------------------------------------
    # Test A: L_RUL gradient routing
    # -------------------------------------------------------------
    pred_rul = trainer.aux_rul(z_prog).squeeze(-1)
    loss_rul = torch.nn.functional.huber_loss(pred_rul, y_rul, delta=1.0)
    loss_rul.backward(retain_graph=True)

    grad_rul = z.grad.clone()
    grad_rul_state = grad_rul[:, :state_dim]
    grad_rul_prog = grad_rul[:, state_dim:]

    norm_rul_state = grad_rul_state.norm().item()
    norm_rul_prog = grad_rul_prog.norm().item()

    print(f"\n[Split {state_dim}/{prog_dim}] L_RUL: ||grad(z_state)||={norm_rul_state:.6f}, ||grad(z_prog)||={norm_rul_prog:.6f}")
    assert norm_rul_state == 0.0, f"Unintended RUL gradient leaked into z_state! Norm: {norm_rul_state}"
    assert norm_rul_prog > 0.0, f"RUL gradient failed to reach z_prog! Norm: {norm_rul_prog}"

    # Reset gradient
    z.grad.zero_()

    # -------------------------------------------------------------
    # Test B: L_VQ gradient routing
    # -------------------------------------------------------------
    out_vq = trainer.vq(z_state)
    terms_vq = trainer.vq.loss_terms(z_state, out_vq)
    loss_vq = terms_vq["commitment"]
    loss_vq.backward(retain_graph=True)

    grad_vq = z.grad.clone()
    grad_vq_state = grad_vq[:, :state_dim]
    grad_vq_prog = grad_vq[:, state_dim:]

    norm_vq_state = grad_vq_state.norm().item()
    norm_vq_prog = grad_vq_prog.norm().item()

    print(f"[Split {state_dim}/{prog_dim}] L_VQ:  ||grad(z_state)||={norm_vq_state:.6f}, ||grad(z_prog)||={norm_vq_prog:.6f}")
    assert norm_vq_state > 0.0, f"VQ commitment gradient failed to reach z_state! Norm: {norm_vq_state}"
    assert norm_vq_prog == 0.0, f"Unintended VQ gradient leaked into z_prog! Norm: {norm_vq_prog}"

    # Reset gradient
    z.grad.zero_()

    # -------------------------------------------------------------
    # Test C: L_recon gradient routing
    # -------------------------------------------------------------
    z_recon = torch.cat([out_vq.straight_through, z_prog], dim=-1)
    X_hat = trainer.decoder(z_recon)
    loss_recon = ((X_hat - X) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    loss_recon.backward()

    grad_recon = z.grad.clone()
    grad_recon_state = grad_recon[:, :state_dim]
    grad_recon_prog = grad_recon[:, state_dim:]

    norm_recon_state = grad_recon_state.norm().item()
    norm_recon_prog = grad_recon_prog.norm().item()

    print(f"[Split {state_dim}/{prog_dim}] L_recon: ||grad(z_state)||={norm_recon_state:.6f}, ||grad(z_prog)||={norm_recon_prog:.6f}")
    assert norm_recon_state > 0.0, f"Recon gradient failed to reach z_state! Norm: {norm_recon_state}"
    assert norm_recon_prog > 0.0, f"Recon gradient failed to reach z_prog! Norm: {norm_recon_prog}"
