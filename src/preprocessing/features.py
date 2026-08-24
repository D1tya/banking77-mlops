from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create the TF-IDF feature extractor.

    We use:
    - unigrams + bigrams
    - minimum document frequency of 2
    - maximum document frequency of 95%
    - sublinear TF scaling
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )

    return vectorizer


def fit_transform(
    train_texts,
    vectorizer: TfidfVectorizer,
):
    """
    Fit the TF-IDF vectorizer on training data
    and transform the training text.
    """

    X_train = vectorizer.fit_transform(train_texts)

    return X_train


def transform(
    texts,
    vectorizer: TfidfVectorizer,
):
    """
    Transform text using an already fitted vectorizer.
    """

    X = vectorizer.transform(texts)

    return X


def save_vectorizer(
    vectorizer: TfidfVectorizer,
    filename: str = "tfidf_vectorizer.joblib",
):
    """
    Save the fitted vectorizer for future inference.
    """

    path = MODEL_DIR / filename

    joblib.dump(vectorizer, path)

    print(f"✓ Vectorizer saved to: {path}")


def load_vectorizer(
    filename: str = "tfidf_vectorizer.joblib",
):
    """
    Load a previously fitted vectorizer.
    """

    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {path}"
        )

    return joblib.load(path)


if __name__ == "__main__":

    from src.data.ingestion import load_data
    from src.validation.validate import run_validation
    from src.preprocessing.preprocess import preprocess_dataset

    # Load
    train_df, test_df, categories = load_data()

    # Validate
    train_df, test_df = run_validation(
        train_df,
        test_df,
        categories,
    )

    # Clean text
    train_df = preprocess_dataset(train_df)
    test_df = preprocess_dataset(test_df)

    # Create vectorizer
    vectorizer = create_tfidf_vectorizer()

    # Fit ONLY on training data
    X_train = fit_transform(
        train_df["cleaned_text"],
        vectorizer,
    )

    # Transform test data
    X_test = transform(
        test_df["cleaned_text"],
        vectorizer,
    )

    # Save vectorizer
    save_vectorizer(vectorizer)

    print("\n========== TF-IDF FEATURES ==========\n")

    print(f"Training feature matrix: {X_train.shape}")
    print(f"Test feature matrix:     {X_test.shape}")

    print(
        f"\nVocabulary size: {len(vectorizer.vocabulary_)}"
    )

    print("\n=====================================\n")
