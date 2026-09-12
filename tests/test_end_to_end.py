"""End-to-end integration test (PRD §29): full pipeline on a small synthetic
dataset — load -> split -> preprocess -> encode -> VQ events -> HSMM ->
RUL/anomaly/health -> schema-valid explanation object.

Also runs the PRD Deliverable E first-experiment skeleton: continuous-AE
baseline vs. VQ+Markov vs. HSMM anomaly scores on synthetic MIMII-like data.
"""
import json

import numpy as np
import pytest
import torch

from egpm.data import UnitDataset, UnitRecord, SplitBuilder, MIMIILoader
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.grammar import ForwardBackward, SegmentalViterbi
from egpm.rul import PhaseTypeRUL
from egpm.anomaly import AnomalyScorer
from egpm.health import HealthHead
from egpm.explanation import ExplanationBuilder, validate_explanation_schema
from egpm.baselines import WindowAutoEncoder
from egpm.evaluation import auroc


def make_degrading_units(n_units=6, T=96, d=4, seed=0):
    """Synthetic runs: multi-frequency healthy oscillation drifting toward
    failure + RUL labels. Windows must be genuinely diverse for VQ event
    discovery (PRD §8's recurring-pattern semantics)."""
    rng = np.random.default_rng(seed)
    ds = UnitDataset()
    for u in range(n_units):
        t = np.arange(T)
        base = np.stack(
            [
                np.sin(0.7 * np.pi * t + rng.uniform(0, 2 * np.pi))
                + 0.5 * np.sin(1.9 * np.pi * t + rng.uniform(0, 2 * np.pi))
                for _ in range(d)
            ],
            axis=1,
        )
        drift = (t / T) ** 2 * (1.0 + 0.3 * u)  # degradation quadratic drift
        signals = base + drift[:, None] + 0.08 * rng.standard_normal((T, d))
        rul = (T - 1) - t.astype(float)
        health = np.minimum(5, 1 + (t // (T // 6)))  # 1..6 ordinal health
        ds.add(UnitRecord(
            unit_id=f"synth_u{u}",
            signals=signals,
            timestamps=t.astype(float),
            rul_labels=rul,
            health_labels=health,
        ))
    return ds


class TestEndToEndPipeline:
    def test_full_pipeline_produces_schema_valid_explanation(self):
        """PRD §29: 'full pipeline runs on a small synthetic dataset without
        error and produces a schema-valid explanation object.'"""
        # 1. data + unit-level split + train-only normalization
        ds = make_degrading_units(n_units=6)
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)

        # 2. preprocessing (deterministic)
        prep = Preprocessor(window_length=8, fs_sync=1.0)
        train_seqs = [prep.process(ds[uid], norm_stats=stats)
                      for uid in split.train_unit_ids]
        runs = [s.X for s in train_seqs]
        run_lengths = [len(r) for r in runs]
        windows = np.concatenate(runs)  # [N, W, d]
        masks = np.concatenate([np.ones_like(r) for r in runs])

        # 3-4. staged training: encoder+VQ (M2/M3), then HSMM EM (M4)
        cfg = TrainerConfig(
            window_length=8, embed_dim=8, hidden=16, n_conv_blocks=2,
            n_codes=8, d_max=6, n_states=3, em_max_iter=6,
            stage1_steps=80, batch_size=16, seed=0, log_every=0,
        )
        trainer = Trainer(cfg, n_channels=windows.shape[-1])
        s1 = trainer.train_stage1(torch.as_tensor(windows, dtype=torch.float32),
                                  torch.as_tensor(masks, dtype=torch.float32))
        assert s1.steps == 80
        assert np.isfinite(s1.recon_loss)
        assert s1.effective_size >= 2, "codebook collapsed"

        seqs_per_window = trainer.event_sequences(
            torch.as_tensor(windows, dtype=torch.float32),
            torch.as_tensor(masks, dtype=torch.float32),
        )
        event_runs = trainer.window_runs_to_event_runs(seqs_per_window, run_lengths)
        s2 = trainer.train_stage2(event_runs)
        assert np.isfinite(s2.em_log_likelihood)
        assert trainer.hsmm is not None
        trainer.hsmm.validate()  # §29 invariants post-training

        # 5-8. heads on a held-out run
        v_test = event_runs[0]
        fb = ForwardBackward(trainer.hsmm)
        res = fb.run(v_test)
        vit = SegmentalViterbi(trainer.hsmm).decode(v_test)
        np.testing.assert_allclose(res.state_posterior.sum(axis=1), 1.0, atol=1e-9)

        rul_head = PhaseTypeRUL(trainer.hsmm)
        belief = res.state_posterior[-1]
        rul = rul_head.rul_from_belief(belief)

        scorer = AnomalyScorer()
        anomaly = scorer.score_sequence(trainer.hsmm, v_test, viterbi=vit)

        health_head = HealthHead(n_states=trainer.hsmm.M)
        health = float(health_head(torch.as_tensor(belief, dtype=torch.float32)).item())
        assert 0.0 <= health <= 1.0

        # explanation object (M8)
        obj = ExplanationBuilder(trainer.hsmm).build(
            unit_id="test", v_seq=v_test, viterbi=vit, anomaly=anomaly,
            rul=rul, health_index=health,
            fault_prediction={"class": 0, "probability": 0.5},
        ).to_json_dict()
        errors = validate_explanation_schema(obj)
        assert errors == [], errors
        json.dumps(obj)  # serializable

    def test_mathematical_invariants_ci_block(self):
        """PRD §29 CI invariants, on a trained-model instance."""
        ds = make_degrading_units(n_units=3, T=48)
        prep = Preprocessor(window_length=8)
        runs = [prep.process(r).X for r in ds.records]
        windows = np.concatenate(runs)
        masks = np.ones_like(windows)
        cfg = TrainerConfig(window_length=8, embed_dim=8, hidden=12,
                            n_conv_blocks=2, n_codes=6, d_max=5, n_states=3,
                            em_max_iter=3, stage1_steps=30, batch_size=16,
                            seed=0, log_every=0)
        trainer = Trainer(cfg, n_channels=windows.shape[-1])
        trainer.train_stage1(torch.as_tensor(windows, dtype=torch.float32),
                            torch.as_tensor(masks, dtype=torch.float32))
        seqs = trainer.event_sequences(torch.as_tensor(windows, dtype=torch.float32),
                                       torch.as_tensor(masks, dtype=torch.float32))
        run_seqs = trainer.window_runs_to_event_runs(seqs, [len(r) for r in runs])
        trainer.train_stage2(run_seqs)
        h = trainer.hsmm
        # all distributions sum to 1
        np.testing.assert_allclose(h.pi.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(h.A.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(h.B.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(h.D.pmf.sum(axis=1), 1.0, atol=1e-9)
        # transient diagonal zero; absorbing row identity
        diag = np.diag(h.A).copy()
        diag[h.absorbing] = 0.0
        assert np.abs(diag).max() < 1e-12
        row = h.A[h.absorbing]
        expected = np.zeros(h.M)
        expected[h.absorbing] = 1.0
        np.testing.assert_allclose(row, expected, atol=1e-12)
        # Viterbi <= total likelihood
        for v in run_seqs:
            ll = ForwardBackward(h).forward(v)[1]
            vl = SegmentalViterbi(h).decode(v).log_likelihood
            assert vl <= ll + 1e-9


class TestFirstExperimentSkeleton:
    """PRD Deliverable E: continuous-AE vs VQ-events+Markov vs HSMM on
    MIMII-like synthetic data. This test wires the full comparison; the
    real-data run happens outside CI (dataset-dependent)."""

    def test_baseline_vs_hsmm_anomaly_scores(self):
        ds = MIMIILoader.synthetic_like(n_units=12, frames_per_unit=400,
                                        anomaly_fraction=0.25, seed=1)
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)
        prep = Preprocessor(window_length=40, fs_sync=1.0)
        # split units into normal (train) vs anomalous (eval proxy)
        normal_ids = [r.unit_id for r in ds.records if r.anomaly_labels[0] == 0]
        anom_ids = [r.unit_id for r in ds.records if r.anomaly_labels[0] == 1]

        train_windows = np.concatenate(
            [prep.process(ds[uid], norm_stats=stats).X for uid in normal_ids[:4]]
        )
        masks = np.ones_like(train_windows)
        Xtr = torch.as_tensor(train_windows, dtype=torch.float32)
        Mtr = torch.as_tensor(masks, dtype=torch.float32)

        # baseline: continuous AE reconstruction score
        ae = WindowAutoEncoder(n_channels=Xtr.shape[-1], window_length=Xtr.shape[1])
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        ae.train()
        for _ in range(60):
            opt.zero_grad()
            rec = ae(Xtr, Mtr)
            loss = (((rec - Xtr) ** 2) * Mtr).mean()
            loss.backward()
            opt.step()
        ae.eval()

        # eval scores: normal held-out vs anomalous
        def eval_scores(ids):
            scores = []
            for uid in ids:
                w = torch.as_tensor(prep.process(ds[uid], norm_stats=stats).X,
                                    dtype=torch.float32)
                m = torch.ones_like(w)
                scores.append(ae.anomaly_scores(w, m).mean().item())
            return np.array(scores)

        normal_scores = eval_scores(normal_ids[4:])
        anom_scores = eval_scores(anom_ids)
        # AE must discriminate bursts on this synthetic data
        labels = np.concatenate([np.zeros(len(normal_scores)),
                                 np.ones(len(anom_scores))])
        scores = np.concatenate([normal_scores, anom_scores])
        auc = auroc(scores, labels)
        assert auc > 0.8, f"continuous AE baseline failed to discriminate ({auc:.2f})"

        # HSMM path: encode->VQ->EM->score (sanity that it runs & separates)
        cfg = TrainerConfig(window_length=40, embed_dim=8, hidden=16,
                            n_conv_blocks=2, n_codes=8, d_max=6, n_states=3,
                            em_max_iter=5, stage1_steps=50, batch_size=16,
                            seed=0, log_every=0)
        trainer = Trainer(cfg, n_channels=Xtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        # normal sequences for EM (PRD §10.4: fit on normal only)
        normal_runs = [prep.process(ds[uid], norm_stats=stats).X
                       for uid in normal_ids[:4]]
        seqs_per_win = trainer.event_sequences(Xtr, Mtr)
        event_runs = trainer.window_runs_to_event_runs(
            seqs_per_win, [len(r) for r in normal_runs]
        )
        trainer.train_stage2(event_runs)

        def hsmm_scores(ids):
            out = []
            for uid in ids:
                run = prep.process(ds[uid], norm_stats=stats)
                X = torch.as_tensor(run.X, dtype=torch.float32)
                m = torch.ones_like(X)
                v = np.concatenate([
                    s for s in trainer.event_sequences(X, m)
                ])
                scorer = AnomalyScorer()
                out.append(scorer.score_sequence(trainer.hsmm, v).score.mean())
            return np.array(out)

        h_normal = hsmm_scores(normal_ids[4:])
        h_anom = hsmm_scores(anom_ids)
        h_auc = auroc(np.concatenate([h_normal, h_anom]),
                      np.concatenate([np.zeros(len(h_normal)),
                                      np.ones(len(h_anom))]))
        # the HSMM path should separate at least weakly on this easy synthetic
        assert h_auc > 0.6, f"HSMM anomaly path failed sanity ({h_auc:.2f})"


class TestReproducibility:
    def test_same_seed_same_initialization(self):
        from egpm.utils import set_global_seed
        cfg = TrainerConfig(window_length=8, embed_dim=8, hidden=12,
                            n_conv_blocks=2, n_codes=6, d_max=5, n_states=3,
                            em_max_iter=3, stage1_steps=5, batch_size=8,
                            seed=123, log_every=0)
        t1 = Trainer(cfg, n_channels=2)
        set_global_seed(123)
        t2 = Trainer(cfg, n_channels=2)
        w1 = t1.encoder.proj.weight.detach()
        w2 = t2.encoder.proj.weight.detach()
        torch.testing.assert_close(w1, w2)
