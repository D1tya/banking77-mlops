from pathlib import Path
import time
import joblib

import mlflow
import mlflow.sklearn

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
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DATABASE}"
)

EXPERIMENT_NAME = (
    "banking77-intent-classification"
)


# ============================================================
# MLFLOW SETUP
# ============================================================

def setup_mlflow():
    """
    Configure MLflow using the local SQLite backend.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(X_train, y_train, C=1.0):
    """
    Train Logistic Regression.
    """

    model = LogisticRegression(
        max_iter=1000,
        C=C,
        solver="lbfgs",
        random_state=42,
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start_time

    return model, training_time


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X, y):
    """
    Evaluate a trained model.
    """

    start_time = time.time()

    predictions = model.predict(X)

    inference_time = time.time() - start_time

    accuracy = accuracy_score(
        y,
        predictions,
    )

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
    Save trained model locally.
    """

    path = MODEL_DIR / filename

    joblib.dump(
        model,
        path,
    )

    print(
        f"\n✓ Model saved to: {path}"
    )


def save_vectorizer(
    vectorizer,
    filename="tfidf_vectorizer.joblib",
):
    """
    Save fitted TF-IDF vectorizer locally.
    """

    path = MODEL_DIR / filename

    joblib.dump(
        vectorizer,
        path,
    )

    print(
        f"✓ Vectorizer saved to: {path}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n========================================")
    print("       BANKING77 MODEL TRAINING")
    print("========================================\n")

    # --------------------------------------------------------
    # 1. SETUP MLFLOW
    # --------------------------------------------------------

    setup_mlflow()

    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------

    train_df, test_df, categories = load_data()

    # --------------------------------------------------------
    # 3. VALIDATE DATA
    # --------------------------------------------------------

    train_df, test_df = run_validation(
        train_df,
        test_df,
        categories,
    )

    # --------------------------------------------------------
    # 4. PREPROCESS
    # --------------------------------------------------------

    train_df = preprocess_dataset(
        train_df
    )

    test_df = preprocess_dataset(
        test_df
    )

    # --------------------------------------------------------
    # 5. SEPARATE TEXT AND LABELS
    # --------------------------------------------------------

    X_text = train_df["cleaned_text"]
    y = train_df["category"]

    X_test_text = test_df["cleaned_text"]
    y_test = test_df["category"]

    # --------------------------------------------------------
    # 6. TRAIN / VALIDATION SPLIT
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

    print(
        f"Training samples:   "
        f"{len(X_train_text)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_val_text)}"
    )

    print(
        f"Test samples:       "
        f"{len(X_test_text)}"
    )

    # --------------------------------------------------------
    # 7. TF-IDF
    # --------------------------------------------------------

    vectorizer = create_tfidf_vectorizer()

    # Fit ONLY on training data
    X_train = vectorizer.fit_transform(
        X_train_text
    )

    # Transform validation/test
    X_val = vectorizer.transform(
        X_val_text
    )

    X_test = vectorizer.transform(
        X_test_text
    )

    vocabulary_size = len(
        vectorizer.vocabulary_
    )

    print("\nFeature matrices:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"\nVocabulary size: "
        f"{vocabulary_size}"
    )

    # --------------------------------------------------------
    # 8. START MLFLOW RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="logistic-regression-baseline"
    ):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        C = 1.0

        mlflow.log_param(
            "model",
            "LogisticRegression",
        )

        mlflow.log_param(
            "solver",
            "lbfgs",
        )

        mlflow.log_param(
            "C",
            C,
        )

        mlflow.log_param(
            "max_iter",
            1000,
        )

        mlflow.log_param(
            "ngram_range",
            "(1, 2)",
        )

        mlflow.log_param(
            "min_df",
            2,
        )

        mlflow.log_param(
            "max_df",
            0.95,
        )

        mlflow.log_param(
            "sublinear_tf",
            True,
        )

        mlflow.log_param(
            "max_features",
            50000,
        )

        mlflow.log_param(
            "training_samples",
            len(X_train_text),
        )

        mlflow.log_param(
            "validation_samples",
            len(X_val_text),
        )

        mlflow.log_param(
            "test_samples",
            len(X_test_text),
        )

        mlflow.log_param(
            "num_classes",
            len(categories),
        )

        mlflow.log_param(
            "vocabulary_size",
            vocabulary_size,
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print(
            "\nTraining Logistic Regression..."
        )

        model, training_time = train_model(
            X_train,
            y_train,
            C=C,
        )

        print(
            f"✓ Training completed in "
            f"{training_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print(
            "\nEvaluating model "
            "on validation set..."
        )

        validation_metrics = evaluate_model(
            model,
            X_val,
            y_val,
        )

        # ----------------------------------------------------
        # Log validation metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "val_accuracy",
            validation_metrics["accuracy"],
        )

        mlflow.log_metric(
            "val_weighted_precision",
            validation_metrics[
                "weighted_precision"
            ],
        )

        mlflow.log_metric(
            "val_weighted_recall",
            validation_metrics[
                "weighted_recall"
            ],
        )

        mlflow.log_metric(
            "val_weighted_f1",
            validation_metrics[
                "weighted_f1"
            ],
        )

        mlflow.log_metric(
            "val_macro_precision",
            validation_metrics[
                "macro_precision"
            ],
        )

        mlflow.log_metric(
            "val_macro_recall",
            validation_metrics[
                "macro_recall"
            ],
        )

        mlflow.log_metric(
            "val_macro_f1",
            validation_metrics[
                "macro_f1"
            ],
        )

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        mlflow.log_metric(
            "validation_inference_time_seconds",
            validation_metrics[
                "inference_time"
            ],
        )

        # ----------------------------------------------------
        # Display validation results
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "       VALIDATION RESULTS"
        )

        print(
            "========================================"
        )

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

        print(
            "\n========================================"
        )

        # ----------------------------------------------------
        # Save model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="logistic_regression_model",
        )

        # ----------------------------------------------------
        # Save vectorizer
        # ----------------------------------------------------

        vectorizer_path = (
            MODEL_DIR /
            "tfidf_vectorizer.joblib"
        )

        joblib.dump(
            vectorizer,
            vectorizer_path,
        )

        mlflow.log_artifact(
            str(vectorizer_path),
            artifact_path="vectorizer",
        )

        # ----------------------------------------------------
        # Save model locally
        # ----------------------------------------------------

        save_model(model)

        save_vectorizer(vectorizer)

        # ----------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------

        print(
            "\nEvaluating model on test set..."
        )

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # ----------------------------------------------------
        # Log test metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "test_accuracy",
            test_metrics["accuracy"],
        )

        mlflow.log_metric(
            "test_weighted_f1",
            test_metrics["weighted_f1"],
        )

        mlflow.log_metric(
            "test_macro_f1",
            test_metrics["macro_f1"],
        )

        mlflow.log_metric(
            "test_macro_precision",
            test_metrics[
                "macro_precision"
            ],
        )

        mlflow.log_metric(
            "test_macro_recall",
            test_metrics[
                "macro_recall"
            ],
        )

        # ----------------------------------------------------
        # Display test results
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "          TEST RESULTS"
        )

        print(
            "========================================"
        )

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

        print(
            "\n========================================"
        )

        # ----------------------------------------------------
        # RUN INFORMATION
        # ----------------------------------------------------

        run = mlflow.active_run()

        if run:

            print(
                f"\nMLflow Run ID: "
                f"{run.info.run_id}"
            )

            print(
                f"MLflow Experiment: "
                f"{EXPERIMENT_NAME}"
            )

        print(
            "\n✓ Training pipeline "
            "completed successfully.\n"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
