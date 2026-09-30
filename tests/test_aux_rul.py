import torch
from egpm.training.trainer import Trainer, TrainerConfig


def test_aux_rul_training_step():
    cfg = TrainerConfig(
        window_length=4,
        embed_dim=8,
        hidden=16,
        n_conv_blocks=2,
        n_codes=4,
        aux_rul_lambda=0.1,
        stage1_steps=5,
        batch_size=4,
        seed=42,
    )
    n_windows = 12
    n_channels = 6
    windows = torch.randn(n_windows, 4, n_channels)
    masks = torch.ones_like(windows)
    rul = torch.linspace(50.0, 1.0, n_windows)

    trainer = Trainer(cfg, n_channels=n_channels)
    stats = trainer.train_stage1(windows, masks, rul_labels=rul)

    assert stats.steps == 5
    assert not torch.isnan(torch.tensor(stats.aux_rul_loss))
    # verify aux_rul linear layer has gradients and was updated
    assert trainer.aux_rul.weight.grad is not None
    assert trainer.aux_rul.bias.grad is not None


def test_aux_rul_zero_lambda_backward_compatible():
    cfg = TrainerConfig(
        window_length=4,
        embed_dim=8,
        hidden=16,
        n_conv_blocks=2,
        n_codes=4,
        aux_rul_lambda=0.0,
        stage1_steps=3,
        batch_size=4,
        seed=42,
    )
    n_windows = 8
    n_channels = 6
    windows = torch.randn(n_windows, 4, n_channels)
    masks = torch.ones_like(windows)

    trainer = Trainer(cfg, n_channels=n_channels)
    stats = trainer.train_stage1(windows, masks)

    assert stats.steps == 3
    assert torch.isnan(torch.tensor(stats.aux_rul_loss))


def test_dead_code_revival():
    from egpm.quantization.vq_codebook import VQCodebook
    vq = VQCodebook(embed_dim=8, n_codes=4)
    vq.ema_cluster_size.fill_(5.0)
    # artificially set one code's cluster size to 0.1
    vq.ema_cluster_size[0] = 0.1
    z = torch.randn(10, 8)
    n_rev = vq.revive_dead_codes(z, threshold=1.0)
    assert n_rev == 1
    assert vq.ema_cluster_size[0] == 1.0

