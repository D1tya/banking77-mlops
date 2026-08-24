from pathlib import Path
import time

import mlflow

from sklearn.svm import LinearSVC
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

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


# ============================================================
# MODEL
# ============================================================

def create_svm(C):

    return LinearSVC(
        C=C,
        max_iter=5000,
        random_state=42,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y):

    start_time = time.time()

    predictions = model.predict(X)

    inference_time = (
        time.time() - start_time
    )

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
# MAIN TUNING PIPELINE
# ============================================================

def main():

    print("\n========================================")
    print("       BANKING77 SVM HYPERPARAMETER")
    print("              TUNING")
    print("========================================\n")

    # --------------------------------------------------------
    # 1. SETUP MLFLOW
    # --------------------------------------------------------

    setup_mlflow()

    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------

    train_df, test_df, categories = (
        load_data()
    )

    # --------------------------------------------------------
    # 3. VALIDATE
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
    # 5. TEXT + LABELS
    # --------------------------------------------------------

    X_text = train_df[
        "cleaned_text"
    ]

    y = train_df[
        "category"
    ]

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

    # --------------------------------------------------------
    # 7. TF-IDF
    # --------------------------------------------------------

    vectorizer = (
        create_tfidf_vectorizer()
    )

    X_train = (
        vectorizer.fit_transform(
            X_train_text
        )
    )

    X_val = (
        vectorizer.transform(
            X_val_text
        )
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
        f"\nVocabulary size: "
        f"{vocabulary_size}"
    )

    # --------------------------------------------------------
    # 8. HYPERPARAMETERS
    # --------------------------------------------------------

    C_VALUES = [
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
    ]

    # --------------------------------------------------------
    # 9. RUN EXPERIMENTS
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "        STARTING HYPERPARAMETER"
    )

    print(
        "               SEARCH"
    )

    print(
        "========================================\n"
    )

    best_C = None

    best_macro_f1 = -1.0

    results = []

    for C in C_VALUES:

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Testing SVM with C = {C}"
        )

        print(
            f"----------------------------------------"
        )

        with mlflow.start_run(
            run_name=f"svm-C-{C}"
        ):

            # ------------------------------------------------
            # Parameters
            # ------------------------------------------------

            mlflow.log_param(
                "model",
                "LinearSVC",
            )

            mlflow.log_param(
                "C",
                C,
            )

            mlflow.log_param(
                "max_iter",
                5000,
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
                "num_classes",
                len(categories),
            )

            mlflow.log_param(
                "vocabulary_size",
                vocabulary_size,
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model = create_svm(C)

            start_time = time.time()

            model.fit(
                X_train,
                y_train,
            )

            training_time = (
                time.time() - start_time
            )

            # ------------------------------------------------
            # Evaluate
            # ------------------------------------------------

            metrics = evaluate_model(
                model,
                X_val,
                y_val,
            )

            # ------------------------------------------------
            # Log metrics
            # ------------------------------------------------

            mlflow.log_metric(
                "val_accuracy",
                metrics["accuracy"],
            )

            mlflow.log_metric(
                "val_weighted_precision",
                metrics[
                    "weighted_precision"
                ],
            )

            mlflow.log_metric(
                "val_weighted_recall",
                metrics[
                    "weighted_recall"
                ],
            )

            mlflow.log_metric(
                "val_weighted_f1",
                metrics[
                    "weighted_f1"
                ],
            )

            mlflow.log_metric(
                "val_macro_precision",
                metrics[
                    "macro_precision"
                ],
            )

            mlflow.log_metric(
                "val_macro_recall",
                metrics[
                    "macro_recall"
                ],
            )

            mlflow.log_metric(
                "val_macro_f1",
                metrics[
                    "macro_f1"
                ],
            )

            mlflow.log_metric(
                "training_time_seconds",
                training_time,
            )

            mlflow.log_metric(
                "inference_time_seconds",
                metrics[
                    "inference_time"
                ],
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(
                {
                    "C": C,
                    "macro_f1": metrics[
                        "macro_f1"
                    ],
                    "accuracy": metrics[
                        "accuracy"
                    ],
                    "training_time": (
                        training_time
                    ),
                }
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            print(
                f"Accuracy:   "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Macro F1:   "
                f"{metrics['macro_f1']:.4f}"
            )

            print(
                f"Train time: "
                f"{training_time:.2f}s"
            )

            # ------------------------------------------------
            # Track best model
            # ------------------------------------------------

            if (
                metrics["macro_f1"]
                > best_macro_f1
            ):

                best_macro_f1 = (
                    metrics["macro_f1"]
                )

                best_C = C

    # --------------------------------------------------------
    # 10. RESULTS SUMMARY
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "          TUNING RESULTS"
    )

    print(
        "========================================\n"
    )

    print(
        f"{'C':<10}"
        f"{'Accuracy':<15}"
        f"{'Macro F1':<15}"
        f"{'Train Time':<15}"
    )

    print(
        "-" * 55
    )

    for result in results:

        print(
            f"{result['C']:<10}"
            f"{result['accuracy']:<15.4f}"
            f"{result['macro_f1']:<15.4f}"
            f"{result['training_time']:<15.2f}"
        )

    # --------------------------------------------------------
    # 11. BEST RESULT
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "          TUNING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nBest C: "
        f"{best_C}"
    )

    print(
        f"Best Validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        "\n✓ Hyperparameter tuning completed."
    )

    print(
        "✓ Best C selected using validation Macro F1."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

