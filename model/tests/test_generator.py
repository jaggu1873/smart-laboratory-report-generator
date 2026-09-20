"""Unit tests for model.generator using the deterministic mock backend."""

from __future__ import annotations

import unittest

from model.config import ModelConfig
from model.generator import DEFAULT_SECTIONS, LabReportGenerator
from model.inference import MockModelBackend
from model.schemas import LabReport, ReportStatus, SectionName


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = {
            "title": "Boiling Point of Water",
            "objective": "determine the boiling point of water at sea level",
            "materials": ["beaker", "thermometer", "hot plate"],
            "procedure": ["Fill beaker", "Heat until boiling", "Record temperature"],
            "measurements": [
                {"name": "temperature", "value": 99.8, "unit": "C"},
                {"name": "volume", "value": 250, "unit": "mL"},
            ],
        }
        self.generator = LabReportGenerator(config=ModelConfig(backend="mock"))

    def test_generate_returns_report(self) -> None:
        report = self.generator.generate(self.raw)
        self.assertIsInstance(report, LabReport)
        self.assertEqual(report.status, ReportStatus.GENERATED)
        self.assertEqual(report.title, "Boiling Point of Water")
        self.assertEqual(report.model_id, "mock-lab-v1")

    def test_all_default_sections_present(self) -> None:
        report = self.generator.generate(self.raw)
        names = [s.name for s in report.sections]
        self.assertEqual(names, list(DEFAULT_SECTIONS))

    def test_sections_have_content(self) -> None:
        report = self.generator.generate(self.raw)
        for section in report.sections:
            self.assertTrue(section.content.strip())

    def test_results_section_lists_measurements(self) -> None:
        report = self.generator.generate(self.raw)
        results = report.get_section(SectionName.RESULTS)
        self.assertIsNotNone(results)
        assert results is not None
        self.assertIn("temperature", results.content)
        self.assertIn("99.8", results.content)

    def test_measurements_carried_through(self) -> None:
        report = self.generator.generate(self.raw)
        self.assertEqual(len(report.measurements), 2)

    def test_deterministic_output(self) -> None:
        r1 = self.generator.generate(self.raw)
        r2 = self.generator.generate(self.raw)
        self.assertEqual(
            [s.content for s in r1.sections],
            [s.content for s in r2.sections],
        )

    def test_default_backend_is_mock(self) -> None:
        self.assertIsInstance(self.generator.backend, MockModelBackend)

    def test_injected_backend_is_used(self) -> None:
        backend = MockModelBackend(ModelConfig(backend="mock", model_id="custom-id"))
        gen = LabReportGenerator(backend=backend)
        report = gen.generate(self.raw)
        self.assertEqual(report.model_id, "custom-id")


if __name__ == "__main__":
    unittest.main()
