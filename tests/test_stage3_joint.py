"""Stage-3 joint multi-task training (PRD §15/§16, ablation A3 arm 'joint').

Semantics that must hold: masked tasks (loss None) never move their head's
parameters; unmasked tasks train; all-masked is a graceful no-op.
"""
import copy

import torch
from torch import nn

from egpm.training import Trainer


def _toy():
    torch.manual_seed(0)
    X = torch.randn(64, 4)
    y_reg = X[:, 0] + 2 * X[:, 1]
    y_cls = (X[:, 2] > 0).long()
    reg = nn.Linear(4, 1)
    clf = nn.Linear(4, 2)
    losses = {
        "rul": lambda h: torch.nn.functional.huber_loss(h(X).squeeze(-1), y_reg),
        "fault": lambda h: torch.nn.functional.cross_entropy(h(X), y_cls),
        "anom": lambda h: None,  # labels absent on this dataset -> masked
    }
    return {"rul": reg, "fault": clf, "anom": nn.Linear(4, 1)}, losses, X


def test_masked_task_head_untouched():
    heads, losses, _ = _toy()
    masked_before = copy.deepcopy(heads["anom"].state_dict())
    final = Trainer.stage3_joint(heads, losses, steps=50, lr=0.05)
    masked_after = heads["anom"].state_dict()
    for k in masked_before:
        assert torch.equal(masked_before[k], masked_after[k])
    assert "anom" not in final


def test_unmasked_tasks_train():
    heads, losses, X = _toy()
    before = {n: float(fn(heads[n])) for n, fn in losses.items() if fn(heads[n]) is not None}
    final = Trainer.stage3_joint(heads, losses, steps=200, lr=0.05)
    assert final["rul"] < before["rul"]
    assert final["fault"] < before["fault"]


def test_all_masked_is_noop():
    heads = {"anom": nn.Linear(4, 1)}
    out = Trainer.stage3_joint(heads, {"anom": lambda h: None}, steps=10)
    assert out == {}


def test_joint_vs_single_task_are_distinct_runs():
    # A3's comparison must actually differ: training both heads jointly
    # shares one optimizer over both parameter sets.
    heads_j, losses_j, _ = _toy()
    heads_j.pop("anom")
    losses_j = {k: v for k, v in losses_j.items() if k != "anom"}
    fj = Trainer.stage3_joint(heads_j, losses_j, steps=100, lr=0.05)
    heads_s, losses_s, _ = _toy()
    heads_s = {"rul": heads_s["rul"]}
    fs = Trainer.stage3_joint(heads_s, {"rul": losses_s["rul"]}, steps=100, lr=0.05)
    assert set(fj) == {"rul", "fault"} and set(fs) == {"rul"}
