"""Clean the raw Wine Quality data and create the binary target."""

from pathlib import Path

import pandas as pd
import yaml

EXPECTED_COLUMNS = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
)


def validate_data(data: pd.DataFrame) -> None:
    """Validate the raw Wine Quality columns, values, and basic ranges."""
    issues = []
    missing = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    unexpected = [column for column in data.columns if column not in EXPECTED_COLUMNS]
    if missing:
        issues.append(f"missing expected columns: {', '.join(missing)}")
    if unexpected:
        issues.append(f"unexpected columns: {', '.join(map(str, unexpected))}")

    null_columns = {
        column: int(data[column].isna().sum())
        for column in EXPECTED_COLUMNS
        if column in data.columns and data[column].isna().any()
    }
    if null_columns:
        details = ", ".join(
            f"{column} ({count})" for column, count in null_columns.items()
        )
        issues.append(f"null values in columns: {details}")

    numeric_columns = {}
    non_numeric = []
    for column in EXPECTED_COLUMNS:
        if column not in data.columns:
            continue
        if pd.api.types.is_numeric_dtype(data[column].dtype):
            numeric_columns[column] = data[column]
        else:
            non_numeric.append(column)
    if non_numeric:
        issues.append(f"non-numeric columns: {', '.join(non_numeric)}")

    for column, values in numeric_columns.items():
        if column == "pH":
            lower, upper = 0, 14
        elif column == "quality":
            lower, upper = 0, 10
        else:
            lower, upper = 0, None
        below = values < lower
        above = values > upper if upper is not None else pd.Series(False, index=values.index)
        if below.any() or above.any():
            bounds = f"{lower} to {upper}" if upper is not None else f">= {lower}"
            issues.append(f"{column} values must be {bounds}")

    if issues:
        raise ValueError("Invalid raw Wine Quality data: " + "; ".join(issues))


def main() -> None:
    """Deduplicate the source data and save the thresholded classification set."""
    project_root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load((project_root / "params.yaml").read_text(encoding="utf-8"))
    data_params = params["data"]
    raw_path = Path(data_params["raw_path"])
    if not raw_path.is_absolute():
        raw_path = project_root / raw_path
    processed_dir = Path(data_params["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    # Load using the source dataset's semicolon delimiter before counting rows.
    data = pd.read_csv(raw_path, sep=";")
    validate_data(data)
    print("Data validation passed")
    rows_before = len(data)
    data = data.drop_duplicates().copy()
    rows_after = len(data)

    threshold = data_params["label_threshold"]
    data["good"] = (data["quality"] >= threshold).astype(int)
    data = data.drop(columns="quality")

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "clean.csv"
    data.to_csv(output_path, index=False)

    print(f"Rows before deduplication: {rows_before}")
    print(f"Rows after deduplication: {rows_after}")
    print("Label balance (good):")
    print(data["good"].value_counts().sort_index().to_string())
    print(f"Saved prepared data to {output_path}")


if __name__ == "__main__":
    main()