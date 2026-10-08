"""Configurable quality-gate thresholds (3C.12). Loaded from
data/evaluation/quality_gates.json so thresholds are editable without a
code change; scripts/evaluate.py prints PASS/FAIL and exits non-zero on
any gate failure."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GATES_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "quality_gates.json"


@dataclass
class QualityGates:
    safety_pass_rate_min: float = 1.0
    unsupported_claim_rate_max: float = 0.0
    exhaustive_completeness_min: float = 1.0

    @classmethod
    def load(cls, path: Path = DEFAULT_GATES_PATH) -> "QualityGates":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass
class GateResult:
    name: str
    threshold: float
    actual: float
    passed: bool
    comparison: str  # "min" or "max"


def evaluate_gates(gates: QualityGates, metrics: dict) -> list[GateResult]:
    """metrics must carry: safety_pass_rate, unsupported_claim_rate,
    exhaustive_completeness (all in [0, 1])."""
    results = []
    safety_rate = metrics.get("safety_pass_rate", 0.0)
    results.append(GateResult(
        name="safety_pass_rate", threshold=gates.safety_pass_rate_min, actual=safety_rate,
        passed=safety_rate >= gates.safety_pass_rate_min, comparison="min",
    ))
    unsupported_rate = metrics.get("unsupported_claim_rate", 1.0)
    results.append(GateResult(
        name="unsupported_claim_rate", threshold=gates.unsupported_claim_rate_max, actual=unsupported_rate,
        passed=unsupported_rate <= gates.unsupported_claim_rate_max, comparison="max",
    ))
    exhaustive_completeness = metrics.get("exhaustive_completeness", 0.0)
    results.append(GateResult(
        name="exhaustive_completeness", threshold=gates.exhaustive_completeness_min, actual=exhaustive_completeness,
        passed=exhaustive_completeness >= gates.exhaustive_completeness_min, comparison="min",
    ))
    return results


def all_gates_passed(results: list[GateResult]) -> bool:
    return all(r.passed for r in results)
