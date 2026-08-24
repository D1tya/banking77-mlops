import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Basic text normalization for Banking77.

    Steps:
    - Convert to lowercase
    - Remove extra whitespace
    - Remove unnecessary punctuation
    """

    text = text.lower()

    # Remove punctuation while keeping letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to a Banking77 dataframe.

    Returns a copy of the dataframe with a cleaned_text column.
    """

    processed_df = df.copy()

    processed_df["cleaned_text"] = (
        processed_df["text"]
        .astype(str)
        .apply(clean_text)
    )

    return processed_df


if __name__ == "__main__":

    from src.data.ingestion import load_data
    from src.validation.validate import run_validation

    # Load data
    train_df, test_df, categories = load_data()

    # Validate data
    train_df, test_df = run_validation(
        train_df,
        test_df,
        categories,
    )

    # Preprocess
    train_processed = preprocess_dataset(train_df)
    test_processed = preprocess_dataset(test_df)

    print("\n========== PREPROCESSING ==========\n")

    print("Original:")
    print(train_df["text"].iloc[0])

    print("\nCleaned:")
    print(train_processed["cleaned_text"].iloc[0])

    print("\nTraining shape:")
    print(train_processed.shape)

    print("\nTest shape:")
    print(test_processed.shape)

    print("\n===================================\n")
