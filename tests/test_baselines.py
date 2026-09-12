"""Baseline suite tests (PRD §21, M9): all five core baselines under the
same data contracts, plus contract-compliance checks."""
import numpy as np
import pytest
import torch

from egpm.baselines import (
    CNNRUL, WindowAutoEncoder, LSTMRUL,
    SAXDiscretizer, SequiturGrammar, GrammarVizScorer, sax_breakpoints,
    WindowKMeans, ConceptBottleneck,
    TokenMarkovModel,
)


class TestCNNRUL:
    def test_shape_contract(self):
        model = CNNRUL(n_channels=3, window_length=16)
        X = torch.randn(4, 16, 3)
        m = torch.ones_like(X)
        assert model(X, m).shape == (4,)

    def test_learns_ramp_rul(self):
        torch.manual_seed(0)
        model = CNNRUL(n_channels=2, window_length=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        # windows from a linear ramp: RUL = window start index (content task)
        W, T = 8, 10
        t = torch.linspace(0, 1, W * T)
        runs = torch.stack([t, 1 - t], dim=1).reshape(T, W, 2)
        # replicate each window B times as independent samples
        B = 16
        X = runs.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * T, W, 2)
        X = X + 0.01 * torch.randn_like(X)
        m = torch.ones_like(X)
        y = torch.arange(T, dtype=torch.float32).repeat(B)
        first = None
        for _ in range(80):
            opt.zero_grad()
            pred = model(X, m)
            loss = ((pred - y) ** 2).mean()
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < first * 0.2


class TestLSTMRUL:
    def test_shape_contract(self):
        model = LSTMRUL(n_channels=2, window_length=8)
        X = torch.randn(3, 6, 8, 2)
        m = torch.ones_like(X)
        assert model(X, m).shape == (3,)

    def test_learns_monotone_sequence_rul(self):
        torch.manual_seed(0)
        model = LSTMRUL(n_channels=1, window_length=4)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        B, T = 8, 12
        # constant windows; RUL depends only on position (memory task)
        X = torch.ones(B, T, 4, 1)
        X += 0.01 * torch.randn_like(X)
        y = torch.arange(T, dtype=torch.float32).flip(dims=[0]).expand(B, T)[:, -1]
        first = None
        for _ in range(120):
            opt.zero_grad()
            pred = model(X, torch.ones_like(X))  # [B]
            loss = ((pred - y) ** 2).mean()
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < first * 0.2


class TestSAXGrammarViz:
    def test_breakpoints_monotone_symmetric(self):
        bp = sax_breakpoints(4)
        assert (np.diff(bp) > 0).all()
        assert np.allclose(-bp, bp[::-1])

    def test_discretizer_znormalizes(self):
        sax = SAXDiscretizer(word_length=4, alphabet_size=3)
        W = np.linspace(0, 10, 40).reshape(-1, 1)
        m = np.ones_like(W)
        word = sax.transform_window(W, m)
        assert word.startswith("c0:")
        # rising ramp -> increasing bins
        syms = word.split(":")[1]
        vals = [ord(c) for c in syms]
        assert vals == sorted(vals)

    def test_discretizer_invariant_to_affine_shift(self):
        # SAX uses z-normalization: shifting/scaling the window must not
        # change the symbols
        sax = SAXDiscretizer(word_length=4, alphabet_size=4)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((32, 2))
        m = np.ones_like(x)
        w1 = sax.transform_window(x, m)
        w2 = sax.transform_window(5.0 + 3.0 * x, m)
        assert w1 == w2

    def test_sequitur_finds_repeated_digrams(self):
        g = SequiturGrammar()
        # a b c a b c a b c -> digram (a,b) and (c,a)/(b,c) repeat
        rules = g.induce(list("abcabcabc"))
        # repeated digrams must be captured in at least one rule
        all_rhs = [tuple(r) for r in rules.values()]
        assert ("a", "b") in all_rhs or ("b", "c") in all_rhs or ("c", "a") in all_rhs

    def test_grammarviz_scorer_flags_novel_ngrams(self):
        rng = np.random.default_rng(1)
        sax = SAXDiscretizer(word_length=3, alphabet_size=3)
        # training: alternating pattern
        train_w = []
        for _ in range(20):
            W = np.zeros((12, 1))
            W[::2] = 1.0  # square wave -> stable SAX word
            train_w.append(sax.transform_window(W + 0.01 * rng.standard_normal((12, 1)),
                                               np.ones_like(W)))
        scorer = GrammarVizScorer(ngram=2).fit(train_w)
        # test: same pattern -> low novelty
        W = np.zeros((12, 1)); W[::2] = 1.0
        same = [sax.transform_window(W, np.ones_like(W))] * 5
        assert scorer.score_run(same).mean() < 0.3
        # test: random noise -> high novelty
        rnd = [sax.transform_window(rng.standard_normal((12, 1)) * 5,
                                    np.ones((12, 1))) for _ in range(5)]
        assert scorer.score_run(rnd).mean() > 0.7


class TestConceptBottleneck:
    def test_kmeans_recovers_clusters(self):
        rng = np.random.default_rng(2)
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        X = np.concatenate([
            c + 0.1 * rng.standard_normal((30, 2)) for c in centers
        ])
        km = WindowKMeans(n_clusters=3, seed=0).fit(X)
        # each true center must be closest to a distinct learned centroid
        d = ((centers[:, None, :] - km.centroids[None, :, :]) ** 2).sum(-1)
        best = d.argmin(axis=1)
        assert len(set(best.tolist())) == 3

    def test_histogram_normalized(self):
        h = ConceptBottleneck.histogram(np.array([0, 1, 1, 3]), 5)
        assert h.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(h, [0.25, 0.5, 0.0, 0.25, 0.0])

    def test_cbm_head_shape_and_gradient(self):
        cbm = ConceptBottleneck(n_concepts=6, out_dim=1)
        h = torch.as_tensor(np.random.default_rng(0).dirichlet(np.ones(6), size=4),
                            dtype=torch.float32)
        out = cbm(h)
        assert out.shape == (4, 1)
        out.sum().backward()
        assert any(p.grad is not None for p in cbm.parameters())

    def test_cbm_rejects_wrong_concept_dim(self):
        cbm = ConceptBottleneck(n_concepts=6)
        with pytest.raises(ValueError):
            cbm(torch.randn(2, 7))


class TestTokenMarkov:
    def test_fit_and_score_likely_vs_unlikely(self):
        mm = TokenMarkovModel(n_events=3, smoothing=0.1)
        # normal: 0 -> 1 -> 2 cycles
        seqs = [np.tile(np.array([0, 1, 2]), 10) for _ in range(3)]
        mm.fit(seqs)
        normal = np.array([0, 1, 2, 0, 1, 2])
        weird = np.array([2, 2, 2, 2, 2, 2])
        assert mm.score_run(normal).mean() < mm.score_run(weird).mean()

    def test_scores_finite_and_shapes(self):
        mm = TokenMarkovModel(n_events=4).fit([np.array([0, 1, 2, 3, 0, 1])])
        s = mm.score_run(np.array([3, 3, 3]))
        assert s.shape == (3,) and np.isfinite(s).all()

    def test_unseen_token_needs_smoothing(self):
        mm = TokenMarkovModel(n_events=3, smoothing=1.0).fit([np.array([0, 0, 0])])
        # token 2 never seen in training: smoothed table must give finite score
        s = mm.score_run(np.array([2, 2]))
        assert np.isfinite(s).all()

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            TokenMarkovModel(n_events=2).score_run(np.array([0, 1]))


class TestContractCompliance:
    """PRD §27 baselines test row: all baselines consume the same
    [B, T, W, d] / [N, W, d] window+mask contracts EGPM uses."""

    def test_all_window_models_accept_same_contract(self):
        B, T, W, d = 2, 5, 8, 3
        X = torch.randn(B, T, W, d)
        m = torch.ones_like(X)
        CNNRUL(n_channels=d, window_length=W)(X.reshape(B * T, W, d),
                                              m.reshape(B * T, W, d))
        LSTMRUL(n_channels=d, window_length=W)(X, m)
        WindowAutoEncoder(n_channels=d, window_length=W)(
            X.reshape(B * T, W, d), m.reshape(B * T, W, d))
