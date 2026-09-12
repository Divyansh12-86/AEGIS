"""EGPM — Event Grammar Probabilistic Model.

Implements the PRD (`Aegis_prd (1).md`): sensor streams -> neural encoder ->
VQ event tokens -> explicit-duration HSMM -> shared state-belief -> RUL /
anomaly / fault / health heads -> structured explanation object.
"""

__version__ = "0.1.0"

STATE_NAMES_DEFAULT = ["startup", "stable", "transition", "abnormal", "degrade", "failure"]
