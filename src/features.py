"""Create leakage-safe, scaled train and test datasets."""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Split the prepared data and fit scaling on training features only."""
    project_root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load((project_root / "params.yaml").read_text(encoding="utf-8"))
    split_params = params["split"]
    processed_dir = project_root / params["data"]["processed_dir"]
    data = pd.read_csv(processed_dir / "clean.csv")

    # Keep the target out of the feature matrix, and stratify both partitions.
    features = data.drop(columns="good")
    labels = data["good"]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=labels,
    )

    # Learn scaling statistics only from the training partition.
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=train_features.columns,
        index=train_features.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=test_features.columns,
        index=test_features.index,
    )
    train_output = train_scaled.assign(good=train_labels)
    test_output = test_scaled.assign(good=test_labels)

    train_output.to_csv(processed_dir / "train.csv", index=False)
    test_output.to_csv(processed_dir / "test.csv", index=False)
    joblib.dump(scaler, processed_dir / "scaler.joblib")
    print(f"Training rows: {len(train_output)}")
    print(f"Test rows: {len(test_output)}")
    print(f"Saved scaled datasets and scaler to {processed_dir}")


if __name__ == "__main__":
    main()