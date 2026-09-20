"""Report-generation orchestration.

Wires the pipeline together: raw input is preprocessed, each report section
is produced by an injected :class:`~model.inference.ModelBackend`, and the
results are assembled into a :class:`~model.schemas.LabReport`.

The generator depends only on the ``ModelBackend`` interface, so the
underlying model can be replaced without changing this file.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import ModelConfig
from .inference import ModelBackend, build_backend
from .preprocessing import preprocess
from .schemas import (
    ExperimentInput,
    LabReport,
    ReportSection,
    ReportStatus,
    SectionName,
)

#: Sections produced, in order, for every generated report.
DEFAULT_SECTIONS: tuple[SectionName, ...] = (
    SectionName.OBJECTIVE,
    SectionName.MATERIALS,
    SectionName.PROCEDURE,
    SectionName.RESULTS,
    SectionName.CONCLUSION,
)


class LabReportGenerator:
    """Generate laboratory reports from raw experiment input.

    Args:
        backend: Inference backend to use. If ``None``, one is built from
            ``config`` via :func:`~model.inference.build_backend`.
        config: Model configuration. Defaults to :meth:`ModelConfig.from_env`.
        sections: Sections to generate, in order. Defaults to
            :data:`DEFAULT_SECTIONS`.
    """

    def __init__(
        self,
        backend: Optional[ModelBackend] = None,
        config: Optional[ModelConfig] = None,
        sections: tuple[SectionName, ...] = DEFAULT_SECTIONS,
    ) -> None:
        self.config = config or ModelConfig.from_env()
        self.backend = backend or build_backend(self.config)
        self.sections = sections

    def generate_from_input(self, experiment: ExperimentInput) -> LabReport:
        """Generate a report from already-preprocessed input.

        Args:
            experiment: Normalized experiment input.

        Returns:
            A :class:`LabReport` with status :attr:`ReportStatus.GENERATED`.
        """
        report_sections = [
            ReportSection(name=name, content=self.backend.generate(name, experiment))
            for name in self.sections
        ]
        return LabReport(
            title=experiment.title,
            sections=report_sections,
            measurements=list(experiment.measurements),
            status=ReportStatus.GENERATED,
            model_id=self.backend.model_id,
        )

    def generate(self, raw: dict[str, Any]) -> LabReport:
        """Preprocess raw input and generate a report end-to-end.

        Args:
            raw: Loosely-structured experiment data.

        Returns:
            The generated :class:`LabReport`.
        """
        experiment = preprocess(raw)
        return self.generate_from_input(experiment)
