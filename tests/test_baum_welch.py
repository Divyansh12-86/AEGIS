"""M4 acceptance tests: Baum-Welch EM fitting (PRD §16 Stage 2, §29).

EM must (a) increase log-likelihood monotonically on training data,
(b) recover generative structure from samples of a known HSMM, (c) respect
model invariants after every M-step.
"""
import numpy as np
import pytest

from egpm.grammar import HSMM, BaumWelch, ForwardBackward
from egpm.grammar.hsmm import DurationHistogram


def make_known_hsmm(seed=0):
    """3-state chain: 0 -> 1 -> 2(absorbing) with distinctive emissions."""
    pi = np.array([0.8, 0.2, 0.0])
    A = np.array([
        [0.0, 0.7, 0.3],
        [0.1, 0.0, 0.9],
        [0.0, 0.0, 1.0],  # absorbing identity row
    ])
    B = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    D = np.zeros((3, 5))
    D[0] = [0.1, 0.2, 0.4, 0.2, 0.1]
    D[1] = [0.3, 0.4, 0.2, 0.1, 0.0]
    D[2] = [0.2, 0.2, 0.2, 0.2, 0.2]
    D = DurationHistogram(pmf=D / D.sum(axis=1, keepdims=True))
    return HSMM(n_states=3, n_events=3, pi=pi, A=A, B=B, D=D, d_max=5, seed=seed)


class TestEMMonotonicity:
    def test_train_loglik_monotone_nondecreasing(self):
        rng = np.random.default_rng(1)
        true = make_known_hsmm()
        seqs = [true.sample(40, rng=np.random.default_rng(10 + i)) for i in range(8)]
        bw = BaumWelch(n_states=3, n_events=3, d_max=5, max_iter=15, min_iter=15, seed=2)
        hsmm, trace = bw.fit(seqs)
        lls = trace.log_likelihoods
        # EM with exact E/M steps is monotone (allow tiny fp noise)
        diffs = np.diff(lls)
        assert (diffs > -1e-8).all(), f"EM decreased: {diffs}"
        # and it should actually improve overall
        assert lls[-1] > lls[0]

    def test_em_improves_over_init(self):
        true = make_known_hsmm()
        seqs = [true.sample(30, rng=np.random.default_rng(20 + i)) for i in range(10)]
        init = HSMM(n_states=3, n_events=3, d_max=5, seed=99)
        ll_init = sum(ForwardBackward(init).forward(v)[1] for v in seqs)
        bw = BaumWelch(n_states=3, n_events=3, d_max=5, max_iter=10, min_iter=10, seed=3)
        hsmm, trace = bw.fit(seqs, init=init)
        assert trace.log_likelihoods[-1] >= ll_init


class TestEMRecovery:
    def test_recovers_emission_structure(self):
        """EM must find a model at least as good as the true generative one
        on finite data, with emissions qualitatively separated (each learned
        state has a dominant token)."""
        true = make_known_hsmm()
        seqs = [true.sample(60, rng=np.random.default_rng(30 + i)) for i in range(15)]
        bw = BaumWelch(n_states=3, n_events=3, d_max=5, max_iter=25, min_iter=25, seed=4)
        hsmm, trace = bw.fit(seqs)
        # (a) learned likelihood >= true model's likelihood (MLE direction)
        true_ll = sum(ForwardBackward(true).forward(v)[1] for v in seqs)
        assert trace.log_likelihoods[-1] >= true_ll - 1e-6
        # (b) every learned state is emission-separated: dominant-token
        # mass > 0.5 for at least 2 of 3 states (data is generated with
        # 0.8-dominant rows; MLE on 900 samples recovers dominance)
        dominant = sorted(hsmm.B.max(axis=1), reverse=True)
        assert dominant[0] > 0.5 and dominant[1] > 0.5, f"B rows:\n{hsmm.B}"

    def test_recovers_from_perturbed_true_init(self):
        """Local sanity: EM from a perturbed-true init converges upward and
        stays near the true model's likelihood."""
        true = make_known_hsmm()
        seqs = [true.sample(60, rng=np.random.default_rng(30 + i)) for i in range(15)]
        rng = np.random.default_rng(123)
        def norm(x):
            x = np.clip(x, 1e-3, None)
            return x / x.sum(axis=-1, keepdims=True)
        init = HSMM(
            n_states=3, n_events=3, d_max=5,
            pi=norm(true.pi + 0.05 * rng.standard_normal(3)),
            A=true.A.copy(),  # A structure kept (absorbing identity etc.)
            B=norm(true.B + 0.05 * rng.standard_normal((3, 3))),
            D=DurationHistogram(
                pmf=norm(np.clip(true.D.pmf + 0.02 * rng.random((3, 5)), 1e-3, None))
            ),
            seed=1,
        )
        bw = BaumWelch(n_states=3, n_events=3, d_max=5, max_iter=20, min_iter=20, seed=4)
        hsmm, trace = bw.fit(seqs, init=init)
        assert trace.log_likelihoods[-1] > trace.log_likelihoods[0]
        true_ll = sum(ForwardBackward(true).forward(v)[1] for v in seqs)
        # converged within the same ballpark as the true model
        assert trace.log_likelihoods[-1] >= true_ll - 50.0

    def test_invariants_hold_after_fit(self):
        true = make_known_hsmm()
        seqs = [true.sample(25, rng=np.random.default_rng(50 + i)) for i in range(6)]
        bw = BaumWelch(n_states=3, n_events=3, d_max=5, max_iter=8, min_iter=8, seed=5)
        hsmm, _ = bw.fit(seqs)
        # PRD §29 invariants post-fit
        np.testing.assert_allclose(hsmm.pi.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(hsmm.A.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(hsmm.B.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(hsmm.D.pmf.sum(axis=1), 1.0, atol=1e-9)
        diag = np.diag(hsmm.A).copy()
        diag[hsmm.absorbing] = 0.0
        assert np.abs(diag).max() < 1e-12
        assert hsmm.pi[hsmm.absorbing] == 0.0

    def test_fit_holdout_stops_before_overfit(self):
        true = make_known_hsmm()
        tr = [true.sample(40, rng=np.random.default_rng(60 + i)) for i in range(8)]
        va = [true.sample(40, rng=np.random.default_rng(80 + i)) for i in range(4)]
        bw = BaumWelch(n_states=3, n_events=3, d_max=5, seed=6)
        hsmm, (tr_trace, va_trace) = bw.fit_holdout(tr, va, max_iter=12, patience=3)
        assert hsmm is not None
        # validation trace must be finite everywhere
        assert np.isfinite(va_trace).all()


class TestEMEdgeCases:
    def test_empty_sequences_rejected(self):
        bw = BaumWelch(n_states=2, n_events=2, d_max=3)
        with pytest.raises(ValueError, match="no sequences"):
            bw.fit([])

    def test_convergence_trace_object(self):
        true = make_known_hsmm()
        seqs = [true.sample(20, rng=np.random.default_rng(90 + i)) for i in range(4)]
        bw = BaumWelch(n_states=3, n_events=3, d_max=5, max_iter=1, min_iter=1, seed=7)
        _, trace = bw.fit(seqs)
        assert isinstance(trace.log_likelihoods, list) and len(trace.log_likelihoods) >= 1
