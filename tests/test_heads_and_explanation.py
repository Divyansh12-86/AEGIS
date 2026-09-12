"""M7/M8 acceptance tests: fault head, health head, explanation schema (PRD §29).

The no-raw-feature-bypass test (PRD §27 fault/ row) verifies the bottleneck
constraint structurally: the primary FaultHead consumes ONLY belief-derived
features; the skip path exists solely behind the A4 flag.
"""
import numpy as np
import pytest
import torch

from egpm.fault import FaultHead
from egpm.health import HealthHead, default_severity_weights
from egpm.explanation import ExplanationBuilder, validate_explanation_schema
from egpm.grammar import HSMM, SegmentalViterbi
from egpm.anomaly import AnomalyScorer
from egpm.rul import PhaseTypeRUL


# ---------------------------------------------------------------------------
# Fault head (PRD §13)
# ---------------------------------------------------------------------------
class TestFaultHead:
    def _make_data(self, B=4, T=10, M=4, seed=0):
        torch.manual_seed(seed)
        beliefs = torch.softmax(torch.randn(B, T, M), dim=-1)
        return beliefs

    def test_output_shape_and_probabilities(self):
        beliefs = self._make_data()
        head = FaultHead(n_states=4, n_classes=3)
        logits = head(beliefs)
        assert logits.shape == (beliefs.shape[0], 3)
        probs = head.predict_proba(beliefs)
        torch.testing.assert_close(
            probs.sum(dim=-1), torch.ones(beliefs.shape[0])
        )

    def test_no_raw_feature_bypass_by_default(self):
        """PRD §27 contract test: primary head has no z/z_hat input path."""
        beliefs = self._make_data()
        head = FaultHead(n_states=4, n_classes=3)
        assert head.use_raw_embedding is False
        # forward does not accept embeddings (raises when passed w/o flag)
        z = torch.randn(beliefs.shape[0], 8)
        with pytest.raises((ValueError, TypeError)):
            head(beliefs, raw_embedding=z)

    def test_a4_ablation_skip_connection_opt_in(self):
        beliefs = self._make_data()
        z = torch.randn(beliefs.shape[0], 8)
        head = FaultHead(n_states=4, n_classes=3, use_raw_embedding=True, embed_dim=8)
        out = head(beliefs, raw_embedding=z)
        assert out.shape == (beliefs.shape[0], 3)
        # skip actually used: embedding changes change logits
        out2 = head(beliefs, raw_embedding=z + 10.0)
        assert not torch.allclose(out, out2)

    def test_duration_statistics_shapes(self):
        beliefs = self._make_data(B=3, T=6, M=4)
        stats = FaultHead.duration_statistics(beliefs)
        assert stats.shape == (3, 4 + 16)

    def test_training_reduces_loss_on_separable_data(self):
        """Belief trajectories with distinct fault signatures are learnable."""
        torch.manual_seed(1)
        M, C, B, T = 4, 2, 32, 12
        head = FaultHead(n_states=M, n_classes=C)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        # class 0: mass early states; class 1: mass late states
        y = torch.randint(0, C, (B,))
        beliefs = torch.zeros(B, T, M)
        for b in range(B):
            focus = [0, 1] if y[b] == 0 else [2, 3]
            beliefs[b, :, focus[0]] = 0.6
            beliefs[b, :, focus[1]] = 0.4
        beliefs += 0.01 * torch.randn_like(beliefs)
        beliefs = torch.softmax(beliefs, dim=-1)
        first = None
        for _ in range(60):
            opt.zero_grad()
            logits = head(beliefs)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < first * 0.3
        acc = (head(beliefs).argmax(-1) == y).float().mean()
        assert acc > 0.9

    def test_gradient_flows(self):
        beliefs = self._make_data()
        head = FaultHead(n_states=4, n_classes=3)
        head(beliefs).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in head.parameters())


# ---------------------------------------------------------------------------
# Health head (PRD §14)
# ---------------------------------------------------------------------------
class TestHealthHead:
    def test_health_in_zero_one(self):
        head = HealthHead(n_states=6)
        beliefs = torch.softmax(torch.randn(8, 6), dim=-1)
        h = head(beliefs)
        assert ((h >= 0) & (h <= 1)).all()

    def test_ordinal_prior_monotone_in_state(self):
        w = default_severity_weights(6)
        assert (np.diff(w) < 0).all()
        assert w[0] == 1.0 and w[-1] == 0.0

    def test_belief_concentrated_on_failure_gives_zero(self):
        head = HealthHead(n_states=4)
        b = torch.zeros(1, 4)
        b[0, 3] = 1.0  # last state = absorbing failure
        torch.testing.assert_close(head(b), torch.zeros(1))

    def test_full_healthy_state_gives_one(self):
        head = HealthHead(n_states=4)
        b = torch.zeros(1, 4)
        b[0, 0] = 1.0
        torch.testing.assert_close(head(b), torch.ones(1))

    def test_mixture_is_linear(self):
        head = HealthHead(n_states=5)
        b1 = torch.zeros(1, 5); b1[0, 0] = 1.0
        b2 = torch.zeros(1, 5); b2[0, 4] = 1.0
        mix = 0.3 * b1 + 0.7 * b2
        want = 0.3 * head(b1) + 0.7 * head(b2)
        torch.testing.assert_close(head(mix), want)

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError):
            HealthHead(n_states=3, weights=np.array([1.5, 0.5, 0.0]))
        with pytest.raises(ValueError):
            HealthHead(n_states=3, weights=np.array([1.0, 0.5]))

    def test_learned_weights_monotonicity_penalty(self):
        head = HealthHead(n_states=4, weights=np.array([0.9, 0.5, 0.7, 0.0]),
                          learn_weights=True)
        # 0.5 -> 0.7 increases toward failure: violation must be penalized
        assert head.monotonicity_penalty() > 0
        good = HealthHead(n_states=4, weights=np.array([0.9, 0.7, 0.4, 0.0]),
                          learn_weights=True)
        assert float(good.monotonicity_penalty()) == pytest.approx(0.0)

    def test_temporal_input_supported(self):
        head = HealthHead(n_states=4)
        beliefs = torch.softmax(torch.randn(2, 7, 4), dim=-1)
        h = head(beliefs)
        assert h.shape == (2, 7)


# ---------------------------------------------------------------------------
# Explanation object (PRD §18)
# ---------------------------------------------------------------------------
def build_full_pipeline_outputs(seed=0):
    """Run the deterministic readouts on a small synthetic run."""
    from egpm.grammar.hsmm import DurationHistogram
    pi = np.array([0.9, 0.1, 0.0])
    A = np.array([[0.0, 0.95, 0.05], [0.95, 0.0, 0.05], [0.0, 0.0, 1.0]])
    B = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.75, 0.20, 0.05],
    ])
    D = np.zeros((3, 6))
    D[0] = [0.05, 0.1, 0.6, 0.15, 0.05, 0.05]
    D[1] = [0.1, 0.5, 0.25, 0.1, 0.05, 0.0]
    D[2] = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    D = DurationHistogram(pmf=D / D.sum(axis=1, keepdims=True))
    hsmm = HSMM(n_states=3, n_events=3, pi=pi, A=A, B=B, D=D, d_max=6, seed=seed)

    rng = np.random.default_rng(seed)
    v = np.zeros(24, dtype=int)
    v[rng.random(24) < 0.08] = 1
    v[15:18] = 2  # anomaly burst

    vit = SegmentalViterbi(hsmm).decode(v)
    scorer = AnomalyScorer()
    anomaly = scorer.score_sequence(hsmm, v, viterbi=vit)
    rul_head = PhaseTypeRUL(hsmm)
    belief = np.zeros(hsmm.M)
    belief[vit.path[-1]] = 1.0
    rul = rul_head.rul_from_belief(belief, n_mc=50,
                                   rng=np.random.default_rng(seed))
    hh = HealthHead(n_states=hsmm.M)
    # belief at last window from Viterbi path (deterministic readout)
    health = float(hh(torch.as_tensor(belief, dtype=torch.float32)).item())
    return hsmm, v, vit, anomaly, rul, health


class TestExplanationBuilder:
    def test_builds_schema_valid_object(self):
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        builder = ExplanationBuilder(hsmm)
        obj = builder.build(
            unit_id="test_unit",
            v_seq=v,
            viterbi=vit,
            anomaly=anomaly,
            rul=rul,
            health_index=health,
            fault_prediction={"class": 0, "probability": 0.6},
        ).to_json_dict()
        errors = validate_explanation_schema(obj)
        assert errors == [], errors

    def test_all_prd18_fields_present_and_typed(self):
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        obj = ExplanationBuilder(hsmm).build(
            unit_id="u", v_seq=v, viterbi=vit, anomaly=anomaly, rul=rul,
            health_index=health,
        ).to_json_dict()
        assert isinstance(obj["events"], list) and len(obj["events"]) == len(v)
        assert {e["window_index"] for e in obj["events"]} == set(range(len(v)))
        assert isinstance(obj["state_trajectory"], list)
        assert len(obj["state_trajectory"]) == len(v)
        assert isinstance(obj["transition_probabilities"]["probability"], float)
        assert isinstance(obj["duration_estimates"]["mean"], float)
        assert isinstance(obj["anomaly_causes"], list) and obj["anomaly_causes"]
        for cause in obj["anomaly_causes"]:
            assert cause["component"] in ("emission", "transition", "duration")
        assert isinstance(obj["RUL_estimate"]["expected"], float)
        assert obj["RUL_estimate"]["median"] is None or isinstance(
            obj["RUL_estimate"]["median"], float
        )
        assert isinstance(obj["health_index"], float)
        assert 0.0 <= obj["health_index"] <= 1.0

    def test_anomaly_causes_sorted_by_surprisal_desc(self):
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        obj = ExplanationBuilder(hsmm).build(
            unit_id="u", v_seq=v, viterbi=vit, anomaly=anomaly, rul=rul,
            health_index=health,
        ).to_json_dict()
        surprisals = [c["surprisal"] for c in obj["anomaly_causes"]]
        assert surprisals == sorted(surprisals, reverse=True)

    def test_burst_window_is_top_anomaly_cause(self):
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        obj = ExplanationBuilder(hsmm).build(
            unit_id="u", v_seq=v, viterbi=vit, anomaly=anomaly, rul=rul,
            health_index=health,
        ).to_json_dict()
        assert obj["anomaly_causes"][0]["window_index"] in range(15, 18)

    def test_fault_prediction_optional(self):
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        obj = ExplanationBuilder(hsmm).build(
            unit_id="u", v_seq=v, viterbi=vit, anomaly=anomaly, rul=rul,
            health_index=health, fault_prediction=None,
        ).to_json_dict()
        assert obj["fault_prediction"] is None
        assert validate_explanation_schema(obj) == []


class TestSchemaValidator:
    def test_validator_catches_missing_keys(self):
        errors = validate_explanation_schema({})
        assert any("missing keys" in e for e in errors)

    def test_validator_catches_bad_component(self):
        base, *_ = build_full_pipeline_outputs()
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        obj = ExplanationBuilder(hsmm).build(
            unit_id="u", v_seq=v, viterbi=vit, anomaly=anomaly, rul=rul,
            health_index=health,
        ).to_json_dict()
        obj["anomaly_causes"][0]["component"] = "vibes"
        errors = validate_explanation_schema(obj)
        assert any("anomaly_causes" in e for e in errors)

    def test_validator_catches_health_out_of_range(self):
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        obj = ExplanationBuilder(hsmm).build(
            unit_id="u", v_seq=v, viterbi=vit, anomaly=anomaly, rul=rul,
            health_index=health,
        ).to_json_dict()
        obj["health_index"] = 1.5
        assert any("health_index" in e for e in validate_explanation_schema(obj))

    def test_json_serializable(self):
        import json
        hsmm, v, vit, anomaly, rul, health = build_full_pipeline_outputs()
        obj = ExplanationBuilder(hsmm).build(
            unit_id="u", v_seq=v, viterbi=vit, anomaly=anomaly, rul=rul,
            health_index=health,
        ).to_json_dict()
        s = json.dumps(obj)  # must not raise
        assert "RUL_estimate" in s
