"""M5 acceptance tests: corrected phase-type RUL formula (PRD §12, §29).

The closed-form synthetic case (PRD §29): a 2-state chain — one transient
state 0 with deterministic dwell, one absorbing state 1 (failure) — where
E[RUL] is analytically known. Also checks the mid-dwell correction, the
fundamental-matrix semantics, and Monte Carlo agreement.
"""
import numpy as np
import pytest

from egpm.grammar import HSMM
from egpm.grammar.hsmm import DurationHistogram
from egpm.rul import PhaseTypeRUL, RULResult


def two_state_deterministic(dwell: int, d_max: int = 20) -> HSMM:
    """State 0 (transient) with ~deterministic dwell; state 1 absorbing."""
    pmf = np.zeros((2, d_max))
    pmf[0, dwell - 1] = 1.0  # deterministic dwell of `dwell` steps
    # absorbing state's dwell is irrelevant (it never leaves) — geometric
    p = 0.5
    pmf[1] = p * (1 - p) ** np.arange(d_max)
    pmf[1] /= pmf[1].sum()
    A = np.array([[0.0, 1.0], [0.0, 1.0]])  # 0 -> 1 w.p. 1; 1 absorbing identity
    return HSMM(
        n_states=2,
        n_events=2,
        pi=np.array([1.0, 0.0]),
        A=A,
        B=np.array([[0.5, 0.5], [0.5, 0.5]]),
        D=DurationHistogram(pmf=pmf),
        d_max=d_max,
    )


class TestClosedFormTwoState:
    """THE PRD §29 synthetic closed-form test."""

    def test_fresh_entry_matches_deterministic_dwell(self):
        # state 0 has deterministic dwell 5 and ALWAYS goes to absorbing.
        # Fresh entry: E[RUL] = 5 (one full dwell, then absorption).
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        assert rul.expected_rul(state=0, tau_elapsed=0) == pytest.approx(5.0)

    def test_mid_dwell_subtracts_elapsed(self):
        # deterministic dwell 5, already elapsed 3: residual r(3) = 2;
        # formula: (r(3) - mu) + (Phi mu) = (2 - 5) + 5 = 2
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        assert rul.expected_rul(state=0, tau_elapsed=3) == pytest.approx(2.0)

    def test_absorbed_state_has_zero_rul(self):
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        assert rul.expected_rul(state=1) == 0.0

    def test_elapsed_beyond_support_clamps(self):
        # elapsed >= D_max: residual -> 0 (or clamped); must not crash/nan
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        val = rul.expected_rul(state=0, tau_elapsed=100)
        assert np.isfinite(val)
        assert val <= 1e-9 + 0.0  # deterministic dwell exhausted


class TestFundamentalMatrix:
    def test_three_chain_cumulative_time(self):
        """Linear chain 0 -> 1 -> 2(absorbing) with deterministic dwells
        3 and 4: fresh-entry RUL from 0 = 3 + 4 = 7; from 1 = 4."""
        pmf = np.zeros((3, 10))
        pmf[0, 2] = 1.0
        pmf[1, 3] = 1.0
        pmf[2] = 0.5 * 0.5 ** np.arange(10)
        pmf[2] /= pmf[2].sum()
        hsmm = HSMM(
            n_states=3, n_events=2,
            pi=np.array([1.0, 0.0, 0.0]),
            A=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            B=np.full((3, 2), 0.5),
            D=DurationHistogram(pmf=pmf),
            d_max=10,
        )
        rul = PhaseTypeRUL(hsmm)
        assert rul.expected_rul(0, 0) == pytest.approx(7.0)
        assert rul.expected_rul(1, 0) == pytest.approx(4.0)

    def test_self_loop_chain_geometric_series(self):
        """State 0 dwells 2, then with p=0.5 re-enters 0 else absorbs into 1.
        E[RUL] fresh = 2 + 0.5*2 + 0.25*2 + ... = 2/(1-0.5) = 4? NO —
        each cycle is one dwell of 2 with re-entry prob 0.5:
        E = 2 * (1 + 0.5 + 0.25 + ...) = 2 * 2 = 4."""
        pmf = np.zeros((2, 5))
        pmf[0, 1] = 1.0  # dwell 2 (d index 1)
        pmf[1] = 0.5 * 0.5 ** np.arange(5)
        pmf[1] /= pmf[1].sum()
        hsmm = HSMM(
            n_states=2, n_events=2,
            pi=np.array([1.0, 0.0]),
            A=np.array([[0.0, 1.0], [0.0, 1.0]]),
            B=np.full((2, 2), 0.5),
            D=DurationHistogram(pmf=pmf),
            d_max=5,
        )
        # override: make state 0 re-enter itself 50%? — zero-diagonal forbids
        # self-transitions; use a 2-transient-state ping-pong instead:
        # 0 -1.0-> 1, 1 -0.5-> 0 else absorb. Dwell(0)=2, dwell(1)=1.
        # Expected: entry 0: 2 + [1 + 0.5*(2 + [1 + ...])]
        # Let V0 = 2 + E1, E1 = 1 + 0.5*V0  => V0 = 2 + 1 + 0.5 V0 => V0 = 6
        pmf = np.zeros((3, 8))
        pmf[0, 1] = 1.0
        pmf[1, 0] = 1.0
        pmf[2] = 0.5 * 0.5 ** np.arange(8)
        pmf[2] /= pmf[2].sum()
        hsmm = HSMM(
            n_states=3, n_events=2,
            pi=np.array([1.0, 0.0, 0.0]),
            A=np.array([
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.0, 1.0],
            ]),
            B=np.full((3, 2), 0.5),
            D=DurationHistogram(pmf=pmf),
            d_max=8,
        )
        rul = PhaseTypeRUL(hsmm)
        assert rul.expected_rul(0, 0) == pytest.approx(6.0)
        assert rul.expected_rul(1, 0) == pytest.approx(4.0)

    def test_phi_counts_visits_not_time(self):
        # PRD Decision 3: (I-Q)^-1 1 counts TRANSITIONS not TIME — verify
        # our Phi mu (time) differs from Phi 1 (transitions) when dwells != 1
        pmf = np.zeros((2, 8))
        pmf[0, 2] = 1.0  # dwell 3
        pmf[1] = 0.5 * 0.5 ** np.arange(8)
        pmf[1] /= pmf[1].sum()
        hsmm = HSMM(
            n_states=2, n_events=2,
            pi=np.array([1.0, 0.0]),
            A=np.array([[0.0, 1.0], [0.0, 1.0]]),
            B=np.full((2, 2), 0.5),
            D=DurationHistogram(pmf=pmf),
            d_max=8,
        )
        rul = PhaseTypeRUL(hsmm)
        # transitions from 0: exactly 1 (0 -> absorbing). time: 3.
        assert rul.Phi[0, 0] == pytest.approx(1.0)  # visits to state 0
        assert rul.expected_rul(0, 0) == pytest.approx(3.0)


class TestBeliefWeighted:
    def test_belief_mixture_is_linear(self):
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        b_mix = np.array([0.5, 0.5])
        b_0 = np.array([1.0, 0.0])
        b_1 = np.array([0.0, 1.0])
        mix = rul.expected_rul_from_belief(b_mix)
        lin = 0.5 * rul.expected_rul_from_belief(b_0) \
            + 0.5 * rul.expected_rul_from_belief(b_1)
        assert mix == pytest.approx(lin)

    def test_three_state_belief_weighting(self):
        pmf = np.zeros((3, 10))
        pmf[0, 2] = 1.0
        pmf[1, 3] = 1.0
        pmf[2] = 0.5 * 0.5 ** np.arange(10)
        pmf[2] /= pmf[2].sum()
        hsmm = HSMM(
            n_states=3, n_events=2,
            pi=np.array([1.0, 0.0, 0.0]),
            A=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            B=np.full((3, 2), 0.5),
            D=DurationHistogram(pmf=pmf),
            d_max=10,
        )
        rul = PhaseTypeRUL(hsmm)
        belief = np.array([0.25, 0.75, 0.0])
        got = rul.expected_rul_from_belief(belief)
        want = 0.25 * 7.0 + 0.75 * 4.0
        assert got == pytest.approx(want)


class TestMonteCarlo:
    def test_mc_mean_agrees_with_formula(self):
        """MC simulation to absorption must agree with the analytic formula
        (within tolerance) on the deterministic 2-state chain."""
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        belief = np.array([1.0, 0.0])
        sims = rul.simulate_to_absorption(belief, n_trajectories=500, rng=np.random.default_rng(0))
        assert np.mean(sims) == pytest.approx(5.0, abs=0.2)
        assert np.all(sims == 5.0)  # deterministic dwell

    def test_mc_median_and_interval(self):
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        res = rul.rul_from_belief(np.array([1.0, 0.0]), n_mc=300, rng=np.random.default_rng(0))
        assert res.median == pytest.approx(5.0)
        assert res.interval[0] <= 5.0 <= res.interval[1]
        assert isinstance(res, RULResult)

    def test_mc_geometric_chain_mean(self):
        """Ping-pong chain from TestFundamentalMatrix: MC mean ~ 6."""
        pmf = np.zeros((3, 8))
        pmf[0, 1] = 1.0
        pmf[1, 0] = 1.0
        pmf[2] = 0.5 * 0.5 ** np.arange(8)
        pmf[2] /= pmf[2].sum()
        hsmm = HSMM(
            n_states=3, n_events=2,
            pi=np.array([1.0, 0.0, 0.0]),
            A=np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]),
            B=np.full((3, 2), 0.5),
            D=DurationHistogram(pmf=pmf),
            d_max=8,
        )
        rul = PhaseTypeRUL(hsmm)
        sims = rul.simulate_to_absorption(
            np.array([1.0, 0.0, 0.0]), n_trajectories=400, rng=np.random.default_rng(1)
        )
        assert np.mean(sims) == pytest.approx(6.0, abs=0.3)

    def test_mc_with_start_elapsed(self):
        # start mid-dwell: elapsed 3 of 5 -> remaining 2 (+0 for chain end)
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        sims = rul.simulate_to_absorption(
            np.array([1.0, 0.0]),
            n_trajectories=200,
            rng=np.random.default_rng(2),
            start_elapsed=np.array([3, 0]),
        )
        assert np.mean(sims) == pytest.approx(2.0, abs=0.15)


class TestEdgeCases:
    def test_invalid_state_rejected(self):
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        with pytest.raises(ValueError):
            rul.expected_rul(state=7)

    def test_belief_shape_validated(self):
        hsmm = two_state_deterministic(dwell=5)
        rul = PhaseTypeRUL(hsmm)
        with pytest.raises(ValueError):
            rul.expected_rul_from_belief(np.zeros(3))

    def test_all_results_finite(self):
        hsmm = HSMM(n_states=4, n_events=5, d_max=6, seed=0)
        rul = PhaseTypeRUL(hsmm)
        for s in range(4):
            for tau in (0, 2, 6):
                v = rul.expected_rul(s, tau)
                assert np.isfinite(v) and v >= 0
