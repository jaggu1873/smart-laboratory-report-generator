"""Model inference interface.

Defines the :class:`ModelBackend` abstraction that decouples the report
generator from any concrete model. Swapping to a real LLM/ML model later
means implementing a new subclass and registering it in
:func:`build_backend` — no other layer changes.

The only backend shipped today is :class:`MockModelBackend`, which is fully
deterministic and makes no network calls.
"""

from __future__ import annotations

import abc
from pathlib import Path

from .config import ModelConfig
from .schemas import ExperimentInput, SectionName


class ModelBackend(abc.ABC):
    """Abstract inference backend.

    A backend receives a section name plus the structured experiment input
    and returns the textual content for that section. Concrete backends
    encapsulate all model-specific behavior (prompting, API calls, decoding).
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def generate(self, section: SectionName, experiment: ExperimentInput) -> str:
        """Generate textual content for ``section`` from ``experiment``.

        Args:
            section: The report section to produce content for.
            experiment: The normalized experiment input.

        Returns:
            The generated content for the section.
        """
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        """Identifier of the underlying model, for provenance tracking."""
        return self.config.model_id


class MockModelBackend(ModelBackend):
    """Deterministic dummy backend used for development and tests.

    Produces plausible, template-based section text derived directly from
    the experiment input. It never makes a network request and requires no
    API key, so it is safe to run anywhere.
    """

    def generate(self, section: SectionName, experiment: ExperimentInput) -> str:
        if section == SectionName.TITLE:
            return experiment.title

        if section == SectionName.OBJECTIVE:
            return f"The objective of this experiment is to {experiment.objective}."

        if section == SectionName.MATERIALS:
            if not experiment.materials:
                return "No materials were recorded for this experiment."
            return "Materials used: " + ", ".join(experiment.materials) + "."

        if section == SectionName.PROCEDURE:
            if not experiment.procedure:
                return "No procedure steps were recorded."
            steps = [f"{i}. {step}" for i, step in enumerate(experiment.procedure, start=1)]
            return "Procedure:\n" + "\n".join(steps)

        if section == SectionName.RESULTS:
            if not experiment.measurements:
                return "No measurements were recorded."
            lines = [
                f"- {m.name}: {m.value} {m.unit}".rstrip()
                for m in experiment.measurements
            ]
            return "The following measurements were recorded:\n" + "\n".join(lines)

        if section == SectionName.CONCLUSION:
            count = len(experiment.measurements)
            return (
                f"Based on {count} recorded measurement(s), the experiment "
                f"addressing '{experiment.objective}' was completed."
            )

        raise ValueError(f"unsupported section: {section!r}")


class SklearnModelBackend(ModelBackend):
    """Predict wine quality with the trained sklearn classifier.

    Third-party ML libraries are imported only when this backend is selected,
    keeping the default model package usable without sklearn installed.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import joblib

        project_root = Path(__file__).resolve().parents[1]
        model_path = Path(config.sklearn_model_path).expanduser()
        scaler_path = Path(config.sklearn_scaler_path).expanduser()
        if not model_path.is_absolute():
            model_path = project_root / model_path
        if not scaler_path.is_absolute():
            scaler_path = project_root / scaler_path

        self._model = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        feature_names = getattr(self._scaler, "feature_names_in_", None)
        if feature_names is None:
            feature_names = getattr(self._model, "feature_names_in_", None)
        if feature_names is None:
            raise ValueError(
                "the sklearn scaler or model must expose feature_names_in_"
            )

        self._feature_names = tuple(str(name) for name in feature_names)
        self._feature_keys = {
            self._normalize_name(name): name for name in self._feature_names
        }
        if len(self._feature_keys) != len(self._feature_names):
            raise ValueError("model feature names are ambiguous after normalization")
        model_feature_names = getattr(self._model, "feature_names_in_", None)
        if model_feature_names is not None and tuple(
            str(name) for name in model_feature_names
        ) != self._feature_names:
            raise ValueError("model and scaler feature orders do not match")

        self._fallback = MockModelBackend(config)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize feature names for case- and whitespace-insensitive lookup."""
        return " ".join(name.casefold().split())

    def generate(self, section: SectionName, experiment: ExperimentInput) -> str:
        """Return template content and, in the conclusion, a model prediction."""
        if section != SectionName.CONCLUSION:
            return self._fallback.generate(section, experiment)

        import numpy as np
        import pandas as pd

        values_by_name = {
            self._normalize_name(measurement.name): measurement.value
            for measurement in experiment.measurements
        }
        missing = [
            feature_name
            for feature_name in self._feature_names
            if self._normalize_name(feature_name) not in values_by_name
        ]
        invalid = []
        feature_values = {}
        for feature_name in self._feature_names:
            key = self._normalize_name(feature_name)
            if key in values_by_name:
                value = values_by_name[key]
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    invalid.append(feature_name)
                    continue
                if not np.isfinite(numeric_value):
                    invalid.append(feature_name)
                    continue
                feature_values[feature_name] = numeric_value

        unavailable = missing + invalid
        conclusion = self._fallback.generate(section, experiment)
        if unavailable:
            details = ", ".join(unavailable)
            return (
                f"{conclusion}\nPrediction unavailable; missing or invalid "
                f"measurements: {details}."
            )

        feature_frame = pd.DataFrame(
            [[feature_values[name] for name in self._feature_names]],
            columns=list(self._feature_names),
        )
        scaled_values = self._scaler.transform(feature_frame)
        scaled_frame = pd.DataFrame(
            np.asarray(scaled_values),
            columns=list(self._feature_names),
            index=feature_frame.index,
        )
        prediction = self._model.predict(scaled_frame)[0]
        predicted_class_index = int(
            np.flatnonzero(np.asarray(self._model.classes_) == prediction)[0]
        )
        confidence = float(
            np.asarray(self._model.predict_proba(scaled_frame))[0][
                predicted_class_index
            ]
        )
        quality = "Good" if int(prediction) == 1 else "Not good"
        return f"{conclusion}\nPredicted quality: {quality} (confidence {confidence:.2f})."


#: Registry mapping backend names to their classes.
_BACKENDS: dict[str, type[ModelBackend]] = {
    "mock": MockModelBackend,
    "sklearn": SklearnModelBackend,
}


def build_backend(config: ModelConfig) -> ModelBackend:
    """Construct the backend named by ``config.backend``.

    Args:
        config: Model configuration selecting the backend.

    Returns:
        An initialized :class:`ModelBackend`.

    Raises:
        ValueError: If ``config.backend`` is not registered.
    """
    try:
        backend_cls = _BACKENDS[config.backend]
    except KeyError:
        known = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"unknown backend '{config.backend}'; available: {known}"
        ) from None
    return backend_cls(config)
