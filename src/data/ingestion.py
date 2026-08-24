from pathlib import Path
import json
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw data directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data():
    """
    Load the Banking77 training, test, and category files.

    Returns:
        train_df: Training dataset
        test_df: Test dataset
        categories: List of category names
    """

    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"
    categories_path = RAW_DATA_DIR / "categories.json"

    # Verify that all required files exist
    required_files = [
        train_path,
        test_path,
        categories_path
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Load category mapping
    with open(categories_path, "r", encoding="utf-8") as file:
        categories = json.load(file)

    return train_df, test_df, categories


if __name__ == "__main__":

    train_df, test_df, categories = load_data()

    print("\n========== BANKING77 DATASET ==========\n")

    print(f"Training samples : {len(train_df)}")
    print(f"Test samples     : {len(test_df)}")
    print(f"Categories       : {len(categories)}")

    print("\nTraining columns:")
    print(train_df.columns.tolist())

    print("\nTraining dataset:")
    print(train_df.head())

    print("\nCategory examples:")
    print(categories[:10])

    print("\n========================================\n")
