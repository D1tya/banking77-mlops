from pathlib import Path
import time
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.data.ingestion import load_data
from src.validation.validate import run_validation
from src.preprocessing.preprocess import preprocess_dataset
from src.preprocessing.features import create_tfidf_vectorizer


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(X_train, y_train):
    """
    Train the baseline Logistic Regression classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        model: Trained Logistic Regression model.
        training_time: Training duration in seconds.
    """

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    return model, training_time


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X, y):
    """
    Evaluate a trained model.

    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: True labels.

    Returns:
        Dictionary containing evaluation metrics.
    """

    start_time = time.time()

    predictions = model.predict(X)

    inference_time = time.time() - start_time

    # Accuracy
    accuracy = accuracy_score(
        y,
        predictions,
    )

    # Weighted metrics
    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # Macro metrics
    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "inference_time": inference_time,
    }


# ============================================================
# MODEL SAVING
# ============================================================

def save_model(
    model,
    filename="logistic_regression.joblib",
):
    """
    Save the trained model to the models directory.
    """

    path = MODEL_DIR / filename

    joblib.dump(model, path)

    print(f"\n✓ Model saved to: {path}")


def save_vectorizer(
    vectorizer,
    filename="tfidf_vectorizer.joblib",
):
    """
    Save the fitted TF-IDF vectorizer.
    """

    path = MODEL_DIR / filename

    joblib.dump(vectorizer, path)

    print(f"✓ Vectorizer saved to: {path}")


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("\n========================================")
    print("       BANKING77 MODEL TRAINING")
    print("========================================\n")

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    train_df, test_df, categories = load_data()

    # --------------------------------------------------------
    # 2. VALIDATE DATA
    # --------------------------------------------------------

    train_df, test_df = run_validation(
        train_df,
        test_df,
        categories,
    )

    # --------------------------------------------------------
    # 3. PREPROCESS TEXT
    # --------------------------------------------------------

    train_df = preprocess_dataset(train_df)
    test_df = preprocess_dataset(test_df)

    # --------------------------------------------------------
    # 4. SEPARATE FEATURES AND LABELS
    # --------------------------------------------------------

    X_text = train_df["cleaned_text"]
    y = train_df["category"]

    X_test_text = test_df["cleaned_text"]
    y_test = test_df["category"]

    # --------------------------------------------------------
    # 5. TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    (
        X_train_text,
        X_val_text,
        y_train,
        y_val,
    ) = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nDataset split:")
    print(f"Training samples:   {len(X_train_text)}")
    print(f"Validation samples: {len(X_val_text)}")
    print(f"Test samples:       {len(X_test_text)}")

    # --------------------------------------------------------
    # 6. CREATE TF-IDF VECTORIZER
    # --------------------------------------------------------

    vectorizer = create_tfidf_vectorizer()

    # IMPORTANT:
    # Fit TF-IDF ONLY on the training data.
    X_train = vectorizer.fit_transform(
        X_train_text
    )

    # Transform validation data using the
    # already-fitted training vectorizer.
    X_val = vectorizer.transform(
        X_val_text
    )

    # Transform test data using the
    # already-fitted training vectorizer.
    X_test = vectorizer.transform(
        X_test_text
    )

    print("\nFeature matrices:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")

    print(
        f"\nVocabulary size: "
        f"{len(vectorizer.vocabulary_)}"
    )

    # --------------------------------------------------------
    # 7. TRAIN MODEL
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model, training_time = train_model(
        X_train,
        y_train,
    )

    print(
        f"✓ Training completed in "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # 8. VALIDATION EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating model on validation set...")

    validation_metrics = evaluate_model(
        model,
        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # 9. DISPLAY VALIDATION RESULTS
    # --------------------------------------------------------

    print("\n========================================")
    print("       VALIDATION RESULTS")
    print("========================================")

    print(
        f"\nAccuracy:           "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Weighted Precision: "
        f"{validation_metrics['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall:    "
        f"{validation_metrics['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{validation_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision:    "
        f"{validation_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:       "
        f"{validation_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    print(
        f"Inference time:     "
        f"{validation_metrics['inference_time']:.4f} seconds"
    )

    print("\n========================================")

    # --------------------------------------------------------
    # 10. SAVE MODEL AND VECTORIZER
    # --------------------------------------------------------

    save_model(model)

    save_vectorizer(vectorizer)

    # --------------------------------------------------------
    # 11. FINAL TEST EVALUATION
    # --------------------------------------------------------
    #
    # We are displaying this for now so we can verify
    # the pipeline. During model selection, the validation
    # metrics are what we will use to compare models.
    #
    # The test set will NOT be used to choose the best model.
    #

    print("\nEvaluating model on test set...")

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\n========================================")
    print("          TEST RESULTS")
    print("========================================")

    print(
        f"\nAccuracy:           "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Weighted Precision: "
        f"{test_metrics['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall:    "
        f"{test_metrics['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision:    "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:       "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Inference time:     "
        f"{test_metrics['inference_time']:.4f} seconds"
    )

    print("\n========================================")

    print(
        "\n✓ Training pipeline completed successfully.\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
