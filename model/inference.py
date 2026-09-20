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


#: Registry mapping backend names to their classes.
_BACKENDS: dict[str, type[ModelBackend]] = {
    "mock": MockModelBackend,
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
