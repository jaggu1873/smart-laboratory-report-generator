"""Clean the raw Wine Quality data and create the binary target."""

from pathlib import Path

import pandas as pd
import yaml


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