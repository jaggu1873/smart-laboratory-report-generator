# Smart Laboratory Report Generator

## Overview

Smart Laboratory Report Generator is an MLOps project that transforms
loosely-structured experiment data into structured, validated laboratory
reports. The repository currently provides a deterministic, dependency-free
Python foundation (preprocessing, schemas, a pluggable model backend, report
generation, and validation) and is evolving toward a full **machine learning +
MLOps pipeline**.

This README describes the project honestly across three horizons: what is
**implemented today**, what is planned for the **Phase 1 MLOps roadmap**
(development and reproducibility), and the **future AI/report-generation
architecture**. Components that are not yet built are explicitly labelled as
*Planned* or *Future* and are not presented as complete.

## Problem Statement

Laboratory experiment records are often captured as free-form, inconsistent
notes: mixed units, missing sections, ambiguous field names, and no guarantee
of structural completeness. Turning these into reliable, comparable reports is
manual, error-prone, and hard to reproduce. There is a need for a system that:

- normalizes heterogeneous raw input into a consistent structure,
- generates a complete laboratory report,
- validates the report deterministically, and
- does all of this in a **reproducible, versioned, and trackable** way suitable
  for an ML lifecycle.

## Project Objectives

1. Provide a clean, deterministic foundation for ingesting and validating
   laboratory experiment data.
2. Establish reproducible MLOps practices: version control, data versioning,
   and experiment tracking.
3. Introduce data-driven components (EDA, feature engineering, baseline models)
   in a structured Phase 1.
4. Evolve toward an AI-assisted report generator built on a pluggable model
   backend, without rewriting the surrounding pipeline.
5. Keep the system testable, reproducible, and academically documented at every
   phase.

## Proposed ML / MLOps Lifecycle

The project follows a standard MLOps lifecycle, adapted to an academic,
phase-based delivery:

```
Data acquisition → EDA → Feature engineering → Baseline models
      → Experiment tracking (MLflow) + Data versioning (DVC)
      → Model evaluation → Model integration (backend)
      → Report generation → Validation → Iteration
```

Reproducibility (Git, DVC, MLflow) is treated as a first-class concern that
wraps the entire lifecycle rather than a final step.

## Phase 1 – Development & Reproducibility

Phase 1 covers the academic review requirements for development and
reproducibility. Items below are marked with their **current status**; unless a
component is explicitly *Completed*, it is planned and **not yet implemented**
in this repository.

### 1. Dataset acquisition / approval — *Planned / Phase 1*
Identify and obtain approval for a suitable laboratory experiment dataset. No
dataset has been added to the repository yet; no dataset name or source is
claimed at this stage.

### 2. Data exploration (EDA) — *Planned / Phase 1*
Perform exploratory data analysis on the approved dataset (distributions,
missing values, unit consistency, outliers). No EDA has been performed yet.

### 3. Feature engineering — *Planned / Phase 1*
Derive model-ready features from the raw dataset. Not yet implemented; the
current code performs deterministic preprocessing/normalization only, not
learned feature engineering.

### 4. Baseline models — *Planned / Phase 1*
Train simple baseline ML models to establish reference performance. No ML models
have been trained; the only model today is a deterministic mock backend.

### 5. Git repository — *Completed*
The project is under Git version control and hosted on GitHub. This is the one
reproducibility component that is fully in place today.

### 6. DVC dataset versioning — *Planned / Phase 1*
Use DVC to version datasets and pipeline artifacts. DVC is **not** configured in
this repository yet; no DVC remote is defined.

### 7. MLflow experiment tracking — *Planned / Phase 1*
Track experiments, parameters, and metrics with MLflow. MLflow is **not**
integrated yet; no experiments or runs exist.

### 8. Initial model evaluation — *Planned / Phase 1*
Evaluate baseline models against agreed metrics. No evaluation has been run and
no scores are reported, because no ML model exists yet.

## Current Implementation Status

The table below reflects the actual state of the repository.

| Component | Status |
|---|---|
| Git/GitHub | Completed |
| Core Python model foundation | Completed |
| Unit normalization | Completed |
| Deterministic validation | Completed |
| Dataset acquisition | Planned / Phase 1 |
| EDA | Planned / Phase 1 |
| Feature engineering | Planned / Phase 1 |
| Baseline ML models | Planned / Phase 1 |
| DVC | Planned / Phase 1 |
| MLflow | Planned / Phase 1 |
| Initial model evaluation | Planned / Phase 1 |
| Real LLM backend | Future |
| PDF/report export | Future |

What exists and works today (the implemented model foundation):

- **Preprocessing** — normalizes loosely-structured raw input into a consistent
  `ExperimentInput` (whitespace collapsing, unit canonicalization).
- **Structured schemas** — dataclass contracts shared across all pipeline
  stages.
- **Pluggable `ModelBackend`** — a single interface that decouples the pipeline
  from any specific model.
- **`MockModelBackend`** — a deterministic, offline backend requiring no API key
  or network access.
- **Report generation** — orchestrated assembly of a `LabReport` from an
  experiment input.
- **Deterministic validation** — structural and data-level checks with no model
  calls (required sections, non-empty fields, finite numeric values, recognized
  units, no duplicate measurements).
- **Centralized unit registry** — one source of truth for canonical units and
  aliases, shared by normalization and validation.
- **38 passing tests** — the standard-library `unittest` suite currently passes
  all 38 tests.

## Current Architecture

The `model` package is organized into focused, single-responsibility modules:

| Module | Responsibility |
|---|---|
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

Configuration is read from environment variables via `ModelConfig.from_env()`,
with safe defaults so the mock backend runs with no setup. API keys are read
fresh from the environment at call time and are never hardcoded or cached on
config objects.

### Quick start (current foundation)

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

experiment = preprocess(raw)
generator = LabReportGenerator(config=ModelConfig(backend="mock"))
report = generator.generate_from_input(experiment)
result = validate_report(report)
```

## Planned ML Pipeline

The following ML pipeline is **planned for Phase 1** and is not yet implemented.
It will be layered onto the existing foundation:

1. **Data ingestion** — load an approved, DVC-versioned dataset.
2. **EDA** — characterize the data and document findings.
3. **Feature engineering** — build reproducible, model-ready features.
4. **Baseline training** — train simple reference models.
5. **Experiment tracking** — log parameters, metrics, and artifacts to MLflow.
6. **Evaluation** — measure baseline performance against agreed metrics.
7. **Backend integration** — expose the trained model through the existing
   `ModelBackend` interface, so the generation/validation pipeline is unchanged.

## Planned AI / Report-Generation Architecture

The current architecture already anticipates an AI-assisted generator: the
pipeline depends only on the `ModelBackend` interface, so a real model can be
introduced without touching preprocessing, generation, or validation.

**Future** work (beyond Phase 1) includes:

- A **real LLM/ML backend** implementing `ModelBackend` alongside the mock.
- Richer, model-driven report content while retaining deterministic validation
  as a guardrail.
- **PDF/report export** for finished reports.

To add a real backend when ready:

1. Subclass `ModelBackend` in `model/inference.py` and implement `generate`.
2. Register it by name in the backend map.
3. Select it via `LAB_MODEL_BACKEND` or `ModelConfig(backend="...")`.

No other layer needs to change. These items are **Future** and are not present
in the repository today.

## Repository Structure

```
Smart-Laboratory-Report-Generator/
├── model/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # Environment-driven configuration
│   ├── generator.py         # Pipeline orchestration
│   ├── inference.py         # ModelBackend interface + MockModelBackend
│   ├── preprocessing.py     # Raw input normalization
│   ├── schemas.py           # Shared dataclass contracts
│   ├── units.py             # Central unit registry
│   ├── validator.py         # Deterministic validation
│   └── tests/               # unittest suite (38 tests)
│       ├── test_generator.py
│       ├── test_preprocessing.py
│       ├── test_units.py
│       └── test_validator.py
├── README.md
├── LICENSE
└── .gitignore
```

> Directories for datasets, DVC, MLflow, EDA notebooks, and ML models are
> **not yet present** — they will be added during Phase 1.

## Testing

The test suite uses the standard-library `unittest` runner and currently reports
**38 passing tests**:

```bash
python -m unittest discover -s model/tests -v
```

No third-party dependencies are required to run the tests.

## Reproducibility Strategy

Reproducibility is a core objective and is being established incrementally:

- **Deterministic core (today)** — the mock backend and validation produce
  identical output for identical input, so the current pipeline is fully
  reproducible with no external services.
- **Version control (today)** — all code is tracked in Git/GitHub.
- **Data versioning (Phase 1, planned)** — DVC will version datasets and
  pipeline artifacts so experiments can be reproduced against exact data.
- **Experiment tracking (Phase 1, planned)** — MLflow will record parameters,
  metrics, and artifacts for every run.
- **Environment isolation** — the foundation uses the standard library only;
  once ML dependencies are introduced, they will be pinned for reproducible
  environments.

## Technology Stack

**Implemented today**

- Python 3.9+ (uses `from __future__ import annotations`)
- Standard library only (no third-party runtime dependencies)
- `unittest` for testing
- Git / GitHub for version control

**Planned for Phase 1 (not yet installed)**

- DVC — dataset and artifact versioning
- MLflow — experiment tracking
- A data/ML stack (e.g., data-analysis and modelling libraries) to be selected
  once the dataset is approved

**Future**

- A real LLM/ML backend
- PDF/report export tooling

## Development Roadmap

| Phase | Focus | Status |
|---|---|---|
| Foundation | Deterministic core: preprocessing, schemas, backend interface, mock backend, generation, validation, tests | Completed |
| Phase 1 | Development & reproducibility: dataset, EDA, feature engineering, baseline models, DVC, MLflow, initial evaluation | Planned / In progress |
| Future | Real LLM/ML backend integration, model-driven report content, PDF/report export | Future |

## License

Released under the [MIT License](LICENSE).
