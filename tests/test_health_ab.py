"""A1: §14 a/b decision experiment for the health head.

- Perfect embedding → learned weights must recover a monotone-decreasing
  profile matching the ordinal truth (and beats the fixed prior where truth
  isn't exactly evenly spaced).
- Penalty monotonicity test: 0 on a monotone sequence, > 0 on a non-monotone.
- Guard test: fit_learned_weights raises on a non-learnable head.
"""
import numpy as np
import torch

from egpm.health import HealthHead, fit_learned_weights


def _synthetic_beliefs(M=6, n_units=4, T=50, seed=3):
    """Beliefs one-hot on state index = (hs-1); hs monotone-decreasing truth."""
    rng = np.random.default_rng(seed)
    beliefs, hs = [], []
    for _ in range(n_units):
        drift = np.linspace(0, M - 1, T).round().astype(int)
        B = np.zeros((T, M))
        B[np.arange(T), drift] = 1.0
        beliefs.append(B)
        hs.append(drift + 1)
    return beliefs, hs


def test_fit_learned_weights_recovers_monotone_ordinality():
    beliefs, hs = _synthetic_beliefs()
    head = HealthHead(n_states=6, learn_weights=True)
    fit_learned_weights(head, beliefs, hs, steps=400, lr=0.05, mono_lambda=1.0)
    w = head.weights.detach().numpy()
    assert w[0] > w[-1], "must learn strictly decreasing ordering"
    w_np = np.asarray(w)
    assert float(w_np[0]) > 0.5 and float(w_np[-1]) < 0.5
    # per-step health index drops over a unit's run
    with torch.no_grad():
        h = head(torch.as_tensor(beliefs[0])).numpy()
    assert h[0] > h[-1]


def test_monotonicity_penalty_semantics():
    m_head = HealthHead(n_states=4, learn_weights=True)
    m_head.weights = torch.nn.Parameter(torch.tensor([1.0, 0.7, 0.3, 0.0], dtype=torch.float32))
    assert m_head.monotonicity_penalty().item() == 0.0
    m_head.weights.data[2] = 0.9  # break monotone: w2 > w1 now
    assert m_head.monotonicity_penalty().item() > 0.0
    # buffer-based head: penalty is 0 (fixed)
    fixed = HealthHead(n_states=4)
    assert fixed.monotonicity_penalty().item() == 0.0


def test_fit_learned_weights_requires_learnable():
    beliefs, hs = _synthetic_beliefs()
    head = HealthHead(n_states=6)
    try:
        fit_learned_weights(head, beliefs, hs)
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-learnable head")
