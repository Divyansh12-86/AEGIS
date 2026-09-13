"""Synthetic analytic HSMM tests (red-team validation, PRD §12/§29).

Every case has a CLOSED-FORM expected RUL computed by hand, independent of
the implementation. Also regression tests for the three math fixes:
pinv fallback, xi underflow, hazard-based duration surprisal.
"""
import numpy as np
import pytest

from egpm.grammar.hsmm import HSMM, DurationHistogram
from egpm.rul import PhaseTypeRUL
from egpm.anomaly import AnomalyScorer


def _hsmm(pi, A, D_pmf, d_max, n_events=2):
    return HSMM(
        n_states=len(pi), n_events=n_events, pi=pi, A=A,
        B=np.full((len(pi), n_events), 1.0 / n_events),
        D=DurationHistogram(pmf=np.asarray(D_pmf, dtype=float)),
        d_max=d_max,
    )


class TestAnalyticExpectedRUL:
    """Hand-computed E[RUL] on chains with known answers."""

    def test_two_stage_certain_chain(self):
        # 0 --dwell exactly 3--> 1 --dwell exactly 4--> absorb(2).
        # Fresh from 0: 3 + 4 = 7. Mid-dwell tau=2 in state 0: 1 + 4 = 5.
        pmf = np.zeros((3, 10))
        pmf[0, 2] = 1.0
        pmf[1, 3] = 1.0
        pmf[2] = np.ones(10) / 10
        h = _hsmm(np.array([1.0, 0, 0]),
                  np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]]), pmf, 10)
        r = PhaseTypeRUL(h)
        assert r.expected_rul(0, 0) == pytest.approx(7.0)
        assert r.expected_rul(0, 2) == pytest.approx(5.0)
        assert r.expected_rul(1, 0) == pytest.approx(4.0)
        assert r.expected_rul(1, 3) == pytest.approx(1.0)

    def test_branching_expected_value(self):
        # From 0: dwell 2, then 0.5 -> absorb, 0.5 -> state 1 (dwell 6,
        # then absorbs). E[RUL|0] = 2 + 0.5*0 + 0.5*6 = 5.
        pmf = np.zeros((3, 10))
        pmf[0, 1] = 1.0
        pmf[1, 5] = 1.0
        pmf[2] = np.ones(10) / 10
        h = _hsmm(np.array([1.0, 0, 0]),
                  np.array([[0, 0.5, 0.5], [0, 0, 1], [0, 0, 1]]), pmf, 10)
        r = PhaseTypeRUL(h)
        assert r.expected_rul(0, 0) == pytest.approx(5.0)
        assert r.expected_rul(1, 0) == pytest.approx(6.0)

    def test_renewal_geometric_reentry(self):
        # State 0: dwell 2, then p=0.5 re-enters 0 (zero-diagonal chain
        # forbids self-segments, but 0->1->0? no: build 0<->1 both dwell 2,
        # each absorbing w.p. 0.5. E[visits to {0,1}] = 2, each dwell 2
        # => E[RUL] = 4 from either state.
        pmf = np.zeros((3, 10))
        pmf[0, 1] = 1.0
        pmf[1, 1] = 1.0
        pmf[2] = np.ones(10) / 10
        h = _hsmm(np.array([1.0, 0, 0]),
                  np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0, 0, 1]]), pmf, 10)
        r = PhaseTypeRUL(h)
        assert r.expected_rul(0, 0) == pytest.approx(4.0)
        assert r.expected_rul(1, 0) == pytest.approx(4.0)

    def test_belief_mixture_matches_weighted_analytic(self):
        # belief 0.5 on state 0 (E=7), 0.5 on state 1 (E=4) => 5.5.
        pmf = np.zeros((3, 10))
        pmf[0, 2] = 1.0
        pmf[1, 3] = 1.0
        pmf[2] = np.ones(10) / 10
        h = _hsmm(np.array([1.0, 0, 0]),
                  np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]]), pmf, 10)
        r = PhaseTypeRUL(h)
        b = np.array([0.5, 0.5, 0.0])
        assert r.expected_rul_from_belief(b) == pytest.approx(5.5)

    def test_tau_beyond_support_is_zero_residual(self):
        # deterministic dwell 3; tau=3 (fully elapsed): RUL = 0 + next
        # state's contribution. 0->absorb always: E[RUL] = 0.
        pmf = np.zeros((2, 5))
        pmf[0, 2] = 1.0
        pmf[1] = np.ones(5) / 5
        h = _hsmm(np.array([1.0, 0]),
                  np.array([[0, 1], [0, 1]]), pmf, 5)
        r = PhaseTypeRUL(h)
        assert r.expected_rul(0, 3) == pytest.approx(0.0)


class TestDegenerateFits:
    def test_never_absorbing_raises_not_silent_zero(self):
        # regression: pinv fallback returned E[RUL] ~ 0 (silently wrong)
        pmf = np.ones((3, 5)) / 5
        h = _hsmm(np.array([1.0, 0, 0]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), pmf, 5)
        with pytest.raises(ValueError, match="cannot reach the absorbing"):
            PhaseTypeRUL(h)


class TestXiUnderflow:
    def test_em_counts_survive_large_negative_ll(self):
        # regression: log(exp(ll)) underflowed to -inf for ll < -745,
        # nan_to_num patched it into wrong (zero) counts.
        from egpm.grammar import BaumWelch
        rng = np.random.default_rng(0)
        h = HSMM(n_states=3, n_events=8, d_max=5, seed=1)
        seqs = [rng.integers(0, 8, 2000) for _ in range(4)]  # ll << -745
        bw = BaumWelch(n_states=3, n_events=8, d_max=5, max_iter=1, min_iter=1, seed=1)
        counts = bw._e_step(h, seqs)
        assert np.isfinite(counts["ll"])
        for k in ("pi", "eta", "B", "D"):
            assert np.isfinite(counts[k]).all(), f"{k} has non-finite counts"
            assert counts[k].sum() > 0, f"{k} all-zero (underflow regression)"


class TestDurationSurprisalScale:
    def test_surprisal_comparable_across_components(self):
        # M3 red-team verdict: -log S(tau) is CORRECT (survival surprisal).
        # The real calibration question is component SCALE: emission,
        # transition, duration surprisals must be the same order of
        # magnitude on normal data, else one silently dominates the score.
        d_max = 8
        pmf = np.zeros((2, d_max))
        pmf[0, 2] = 0.7; pmf[0, 1] = 0.3  # dwell 2-3
        pmf[1] = np.ones(d_max) / d_max
        h = _hsmm(np.array([1.0, 0]),
                  np.array([[0, 1], [0, 1]]), pmf, d_max)
        rng = np.random.default_rng(0)
        v = rng.integers(0, 2, 40)
        comp = AnomalyScorer().score_sequence(h, v)
        for name in ("emission_surprisal", "transition_surprisal",
                     "duration_surprisal"):
            arr = getattr(comp, name)
            assert np.isfinite(arr).all()
            assert arr.mean() < 30.0, f"{name} dominates (scale mismatch)"
        # a dwell of tau=D_max on a legal-to-long state is penalized but finite
        assert comp.duration_surprisal.max() < 50.0


class TestPosteriorWeightedRUL:
    def test_belief_weighting_beats_map_pathologically(self):
        # belief 0.51 on state 0 (E=10: dwell 1 then forced 1(dwell 9)),
        # 0.49 on state 1 (E=9). MAP picks E=10; weighted ~9.5 — the point
        # is weighted == mixture, NOT the MAP value, when beliefs split.
        pmf = np.zeros((3, 10))
        pmf[0, 0] = 1.0   # dwell 1
        pmf[1, 8] = 1.0   # dwell 9
        pmf[2] = np.ones(10) / 10
        h = _hsmm(np.array([1.0, 0, 0]),
                  np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]]), pmf, 10)
        r = PhaseTypeRUL(h)
        e0, e1 = r.expected_rul(0, 0), r.expected_rul(1, 0)
        assert e0 == pytest.approx(10.0)  # 1 + 9: forced chain through 1
        assert e1 == pytest.approx(9.0)
        belief = np.array([0.51, 0.49, 0.0])
        weighted = r.expected_rul_from_belief(belief)
        assert weighted == pytest.approx(0.51 * e0 + 0.49 * e1)
        # weighted differs from BOTH pure-state answers when beliefs split
        assert weighted != pytest.approx(e0)
        assert weighted != pytest.approx(e1)
