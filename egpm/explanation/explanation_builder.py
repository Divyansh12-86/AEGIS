"""Structured explanation object (PRD §18, M8).

Assembles the deterministic JSON object from pipeline outputs — events,
state trajectory, transition/duration estimates, anomaly causes, RUL,
fault prediction, health index. Schema (PRD §18):

    {
      "events": [{"window_index": int, "event_id": int}],
      "state_trajectory": [{"window_index": int, "state": int, "elapsed_duration": int}],
      "transition_probabilities": {"from_state": int, "to_state": int, "probability": float},
      "duration_estimates": {"state": int, "mean": float, "residual_given_elapsed": float},
      "anomaly_causes": [{"window_index": int, "component": "emission|transition|duration", "surprisal": float}],
      "RUL_estimate": {"expected": float, "median": float|null, "interval": [lo, hi]|null},
      "fault_prediction": {"class": int, "probability": float},
      "health_index": float
    }

This is the primary interpretability mechanism — no LLM in the MVP (PRD §18).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..grammar.hsmm import HSMM
from ..grammar.viterbi import ViterbiResult
from ..anomaly.anomaly_scorer import AnomalyComponents
from ..rul.phase_type_rul import RULResult


@dataclass
class ExplanationObject:
    """Dataclass mirror of the PRD §18 JSON schema."""

    events: List[Dict[str, int]]
    state_trajectory: List[Dict[str, int]]
    transition_probabilities: Dict[str, float]
    duration_estimates: Dict[str, float]
    anomaly_causes: List[Dict[str, Any]]
    RUL_estimate: Dict[str, Any]
    fault_prediction: Optional[Dict[str, Any]]
    health_index: float

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "events": self.events,
            "state_trajectory": self.state_trajectory,
            "transition_probabilities": self.transition_probabilities,
            "duration_estimates": self.duration_estimates,
            "anomaly_causes": self.anomaly_causes,
            "RUL_estimate": self.RUL_estimate,
            "fault_prediction": self.fault_prediction,
            "health_index": self.health_index,
        }


def validate_explanation_schema(obj: Dict[str, Any]) -> List[str]:
    """Schema validation (PRD §27 explanation/ tests). Returns list of
    violations; empty list == valid."""
    errors: List[str] = []
    required_keys = {
        "events", "state_trajectory", "transition_probabilities",
        "duration_estimates", "anomaly_causes", "RUL_estimate",
        "fault_prediction", "health_index",
    }
    missing = required_keys - set(obj.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
        return errors
    # events
    if not isinstance(obj["events"], list) or not all(
        set(e) == {"window_index", "event_id"} for e in obj["events"]
    ):
        errors.append("events: list of {window_index, event_id} required")
    # state_trajectory
    if not isinstance(obj["state_trajectory"], list) or not all(
        set(s) == {"window_index", "state", "elapsed_duration"}
        for s in obj["state_trajectory"]
    ):
        errors.append("state_trajectory: {window_index, state, elapsed_duration} required")
    # transition_probabilities
    tp = obj["transition_probabilities"]
    if not (
        isinstance(tp, dict)
        and {"from_state", "to_state", "probability"} <= set(tp.keys())
    ):
        errors.append("transition_probabilities: from/to/probability required")
    # duration_estimates
    de = obj["duration_estimates"]
    if not (isinstance(de, dict) and {"state", "mean", "residual_given_elapsed"} <= set(de.keys())):
        errors.append("duration_estimates: state/mean/residual_given_elapsed required")
    # anomaly_causes
    if not isinstance(obj["anomaly_causes"], list) or not all(
        set(a) == {"window_index", "component", "surprisal"}
        and a["component"] in ("emission", "transition", "duration")
        for a in obj["anomaly_causes"]
    ):
        errors.append("anomaly_causes: {window_index, component in emission|transition|duration, surprisal}")
    # RUL_estimate
    rul = obj["RUL_estimate"]
    if not (isinstance(rul, dict) and "expected" in rul):
        errors.append("RUL_estimate: 'expected' required")
    elif not isinstance(rul["expected"], (int, float)):
        errors.append("RUL_estimate.expected must be numeric")
    # fault_prediction
    fp = obj["fault_prediction"]
    if fp is not None:
        if not (isinstance(fp, dict) and {"class", "probability"} <= set(fp.keys())):
            errors.append("fault_prediction: {class, probability} or null required")
    # health_index
    hi = obj["health_index"]
    if not isinstance(hi, (int, float)) or not (0.0 <= float(hi) <= 1.0):
        errors.append("health_index must be a float in [0, 1]")
    return errors


class ExplanationBuilder:
    """Assemble the explanation object from one run's pipeline outputs."""

    def __init__(self, hsmm: HSMM):
        self.hsmm = hsmm

    # ------------------------------------------------------------------
    def build(
        self,
        unit_id: str,
        v_seq: np.ndarray,
        viterbi: ViterbiResult,
        anomaly: AnomalyComponents,
        rul: RULResult,
        health_index: float,
        fault_prediction: Optional[Dict[str, Any]] = None,
        belief: Optional[np.ndarray] = None,
        top_k_anomaly_windows: int = 5,
        current_state: Optional[int] = None,
        current_elapsed: Optional[int] = None,
    ) -> ExplanationObject:
        """Build the PRD §18 object for one run at the final window.

        All inputs are deterministic readouts of the pipeline (PRD §4).
        """
        v_seq = np.asarray(v_seq, dtype=np.int64)
        T = len(v_seq)
        events = [
            {"window_index": int(t), "event_id": int(v_seq[t])} for t in range(T)
        ]
        state_trajectory = [
            {
                "window_index": int(t),
                "state": int(viterbi.path[t]),
                "elapsed_duration": int(viterbi.duration_estimates[t]),
            }
            for t in range(T)
        ]
        # transitions: last Viterbi boundary (or absorbing self-loop)
        s_last = int(viterbi.path[-1])
        s_prev = int(viterbi.path[max(T - 2, 0)]) if T > 1 else s_last
        if s_prev != s_last:
            prob = float(self.hsmm.A[s_prev, s_last])
            transition_probabilities = {
                "from_state": s_prev,
                "to_state": s_last,
                "probability": prob,
            }
        else:
            transition_probabilities = {
                "from_state": s_last,
                "to_state": s_last,
                "probability": float(self.hsmm.A[s_last, s_last]),
            }
        # duration estimates for the current segment's state
        st = current_state if current_state is not None else s_last
        el = int(current_elapsed if current_elapsed is not None
                else viterbi.duration_estimates[-1])
        el = int(np.clip(el, 0, self.hsmm.d_max))
        transient = [i for i in range(self.hsmm.M) if i != self.hsmm.absorbing]
        mean_dwell = float(self.hsmm.D.mean()[st])
        if st in transient:
            ti = transient.index(st)
            R_table = self.hsmm.D.mean_residual(np.zeros(len(transient), dtype=int))
            residual = float(R_table[ti, el])
        else:
            residual = 0.0
        duration_estimates = {
            "state": int(st),
            "mean": mean_dwell,
            "residual_given_elapsed": residual,
        }
        # anomaly causes: top-K windows by surprisal
        order = np.argsort(anomaly.score)[::-1][:top_k_anomaly_windows]
        anomaly_causes = [
            {
                "window_index": int(t),
                "component": anomaly.flagged_component[t],
                "surprisal": float(anomaly.score[t]),
            }
            for t in order
        ]
        RUL_estimate = {
            "expected": float(rul.expected),
            "median": None if rul.median is None else float(rul.median),
            "interval": None if rul.interval is None
            else [float(rul.interval[0]), float(rul.interval[1])],
        }
        return ExplanationObject(
            events=events,
            state_trajectory=state_trajectory,
            transition_probabilities=transition_probabilities,
            duration_estimates=duration_estimates,
            anomaly_causes=anomaly_causes,
            RUL_estimate=RUL_estimate,
            fault_prediction=fault_prediction,
            health_index=float(health_index),
        )
