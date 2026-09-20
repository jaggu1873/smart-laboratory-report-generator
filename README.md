# Smart Laboratory Report Generator

A dependency-free Python library that turns loosely-structured experiment data
into a structured, validated laboratory report. The pipeline is deterministic
by default and ships with an offline mock model, so it runs anywhere with no
API keys and no network access.

The underlying model is pluggable: swapping in a real LLM/ML backend later
means implementing one interface, with no changes to the rest of the pipeline.

## Features

- **Clean pipeline stages** — preprocessing → generation → validation, each
  isolated and independently testable.
- **Deterministic mock backend** — reproducible output for development, tests,
  and CI; no API key required.
- **Centralized unit registry** — a single source of truth for canonical
  laboratory units and their aliases, shared by normalization and validation.
- **Deterministic validation** — structural and data-level checks (required
  sections, non-empty fields, finite numeric values, recognized units, no
  duplicate measurements) with no model calls.
- **Secrets stay out of code** — configuration is read from environment
  variables; API keys are never hardcoded or cached on config objects.
- **Standard library only** — no third-party runtime dependencies.

## Requirements

- Python 3.9 or newer (the code uses `from __future__ import annotations`).

## Installation

Clone the repository and use the `model` package directly:

```bash
git clone https://github.com/<your-org>/Smart-Laboratory-Report-Generator.git
cd Smart-Laboratory-Report-Generator
```

No dependencies need to be installed to run the library or its tests.

## Quick start

```python
from model import LabReportGenerator, ModelConfig, preprocess, validate_report

raw = {
    "title": "Boiling Point of Water",
    "objective": "determine the boiling point of water at sea level",
    "materials": ["beaker", "thermometer", "hot plate"],
    "procedure": ["Fill beaker", "Heat until boiling", "Record temperature"],
    "measurements": [
        {"name": "temperature", "value": 99.8, "unit": "Celsius"},
        {"name": "volume", "value": 250, "unit": "ml"},
    ],
}

# 1. Normalize raw input (collapses whitespace, canonicalizes units).
experiment = preprocess(raw)

# 2. Generate a report using the deterministic mock backend.
generator = LabReportGenerator(config=ModelConfig(backend="mock"))
report = generator.generate_from_input(experiment)

# 3. Validate the generated report.
result = validate_report(report)
print(report.status.value)   # "generated"
print(result.is_valid)       # True
```

`LabReportGenerator.generate(raw)` runs preprocessing and generation together
if you want the full pipeline in one call.

## Architecture

The `model` package is organized into focused, single-responsibility modules:

| Module | Responsibility |
| --- | --- |
| `model/schemas.py` | Dataclass contracts shared across pipeline stages. |
| `model/units.py` | Central registry of canonical units and their aliases. |
| `model/preprocessing.py` | Cleans and normalizes raw input into `ExperimentInput`. |
| `model/inference.py` | `ModelBackend` interface and the deterministic `MockModelBackend`. |
| `model/generator.py` | Orchestrates preprocessing, generation, and assembly. |
| `model/validator.py` | Deterministic structural and data validation. |
| `model/config.py` | Environment-driven configuration; reads API keys at call time. |

Data flows in one direction:

```
raw dict → preprocess() → ExperimentInput → LabReportGenerator → LabReport → validate_report() → ValidationResult
```

## Configuration

Configuration is read from environment variables via `ModelConfig.from_env()`.
All values have safe defaults, so nothing is required to run the mock backend.

| Environment variable | Default | Description |
| --- | --- | --- |
| `LAB_MODEL_BACKEND` | `mock` | Inference backend to build. |
| `LAB_MODEL_ID` | `mock-lab-v1` | Identifier of the underlying model. |
| `LAB_MODEL_TEMPERATURE` | `0.2` | Sampling temperature for generation. |
| `LAB_MODEL_MAX_TOKENS` | `1024` | Maximum tokens a backend may produce. |
| `LAB_MODEL_TIMEOUT` | `30` | Per-request timeout (seconds) for a real backend. |
| `LAB_MODEL_API_KEY_ENV` | `LAB_MODEL_API_KEY` | Name of the variable holding the API key for a real backend. |

API keys are read fresh from the environment on every call and are never stored
on the config object or committed to the repository.

## Running the tests

The test suite uses the standard-library `unittest` runner:

```bash
python -m unittest discover -s model/tests -v
```

## Extending with a real backend

To add a real model backend:

1. Subclass `ModelBackend` in `model/inference.py` and implement `generate`.
2. Register it in the `_BACKENDS` map by name.
3. Select it via `LAB_MODEL_BACKEND` or `ModelConfig(backend="...")`.

No other layer needs to change.

## License

Released under the [MIT License](LICENSE).
