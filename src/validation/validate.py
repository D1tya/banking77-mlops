from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Expected Banking77 schema
TRAIN_SCHEMA = pa.DataFrameSchema(
    {
        "text": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.strip().str.len() > 0)
            ],
        ),
        "category": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.strip().str.len() > 0)
            ],
        ),
    },
    strict=True,
    coerce=True,
)


TEST_SCHEMA = pa.DataFrameSchema(
    {
        "text": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.strip().str.len() > 0)
            ],
        ),
        "category": Column(
            str,
            nullable=False,
            checks=[
                Check(lambda s: s.str.strip().str.len() > 0)
            ],
        ),
    },
    strict=True,
    coerce=True,
)


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Validate a Banking77 dataset against the expected schema.

    Args:
        df: Dataset to validate.
        dataset_name: Name used for logging/error messages.

    Returns:
        Validated dataframe.
    """

    print(f"\nValidating {dataset_name} dataset...")

    schema = TRAIN_SCHEMA if dataset_name == "train" else TEST_SCHEMA

    validated_df = schema.validate(df)

    print(f"✓ {dataset_name} schema validation passed")

    return validated_df


def validate_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categories: list,
) -> None:
    """
    Validate that all labels belong to the Banking77 category list.
    """

    valid_categories = set(categories)

    train_categories = set(train_df["category"].unique())
    test_categories = set(test_df["category"].unique())

    invalid_train = train_categories - valid_categories
    invalid_test = test_categories - valid_categories

    if invalid_train:
        raise ValueError(
            f"Invalid categories found in training data: {invalid_train}"
        )

    if invalid_test:
        raise ValueError(
            f"Invalid categories found in test data: {invalid_test}"
        )

    print("✓ All categories are valid")


def validate_duplicates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Check for duplicate text entries.
    """

    train_duplicates = train_df["text"].duplicated().sum()
    test_duplicates = test_df["text"].duplicated().sum()

    print(f"Duplicate training texts: {train_duplicates}")
    print(f"Duplicate test texts: {test_duplicates}")


def validate_class_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Check that all expected categories are represented.
    """

    train_classes = set(train_df["category"].unique())
    test_classes = set(test_df["category"].unique())

    print(f"Training classes: {len(train_classes)}")
    print(f"Test classes: {len(test_classes)}")

    if len(train_classes) != len(test_classes):
        raise ValueError(
            "Training and test datasets do not contain the same number "
            "of categories."
        )

    print("✓ Class distribution validation passed")


def run_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categories: list,
):
    """
    Run the complete Banking77 validation pipeline.
    """

    train_df = validate_dataset(train_df, "train")
    test_df = validate_dataset(test_df, "test")

    validate_categories(
        train_df,
        test_df,
        categories,
    )

    validate_duplicates(
        train_df,
        test_df,
    )

    validate_class_distribution(
        train_df,
        test_df,
    )

    print("\n========================================")
    print("✓ ALL DATA VALIDATION CHECKS PASSED")
    print("========================================\n")

    return train_df, test_df


if __name__ == "__main__":

    # Import ingestion function
    from src.data.ingestion import load_data

    train_df, test_df, categories = load_data()

    run_validation(
        train_df,
        test_df,
        categories,
    )
