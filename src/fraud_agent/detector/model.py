"""Hybrid detector: Isolation Forest (unknown unknowns) + rules (known patterns)."""
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from fraud_agent.detector.features import FEATURE_NAMES, rule_hits


@dataclass
class Detector:
    model: IsolationForest
    threshold: float

    def score(self, f: dict[str, float]) -> tuple[bool, float, list[str]]:
        return self.score_many([f])[0]

    def score_many(self, fs: list[dict[str, float]]) -> list[tuple[bool, float, list[str]]]:
        """Vectorised scoring: one decision_function call for the whole batch."""
        X = np.array([[f[k] for k in FEATURE_NAMES] for f in fs])
        raws = self.model.decision_function(X)          # < 0 => anomalous
        return [self._combine(f, float(raw)) for f, raw in zip(fs, raws)]

    def _combine(self, f: dict[str, float], raw: float) -> tuple[bool, float, list[str]]:
        iso_score = float(1 / (1 + np.exp(raw * 20)))   # squash to 0..1, higher = worse
        reasons = rule_hits(f)
        if raw < self.threshold:
            reasons.append(f"isolation-forest outlier (decision={raw:.3f})")
        score = max(iso_score, min(1.0, 0.5 + 0.15 * len(reasons))) if reasons else iso_score
        return bool(reasons), round(score, 3), reasons

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str) -> "Detector":
        d = joblib.load(path)
        return cls(model=d["model"], threshold=d["threshold"])
