from pathlib import Path
import time
import joblib

import mlflow
import mlflow.sklearn

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

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

MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DATABASE}"
)

EXPERIMENT_NAME = (
    "banking77-intent-classification"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

MODEL_TYPE = "LinearSVC"

BEST_C = 0.5

MAX_ITER = 5000


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
# MODEL EVALUATION
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
# MAIN FINAL TRAINING PIPELINE
# ============================================================

def main():

    print("\n========================================")
    print("       BANKING77 FINAL MODEL")
    print("========================================\n")

    print(
        f"Model: {MODEL_TYPE}"
    )

    print(
        f"Selected C: {BEST_C}"
    )

    print(
        "\n========================================"
    )

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

    X_train_text = train_df[
        "cleaned_text"
    ]

    y_train = train_df[
        "category"
    ]

    X_test_text = test_df[
        "cleaned_text"
    ]

    y_test = test_df[
        "category"
    ]

    print("\nDataset:")

    print(
        f"Training samples: "
        f"{len(X_train_text)}"
    )

    print(
        f"Test samples:     "
        f"{len(X_test_text)}"
    )

    print(
        f"Classes:           "
        f"{len(categories)}"
    )

    # --------------------------------------------------------
    # 6. TF-IDF
    # --------------------------------------------------------

    print(
        "\nCreating TF-IDF features..."
    )

    vectorizer = (
        create_tfidf_vectorizer()
    )

    start_time = time.time()

    X_train = (
        vectorizer.fit_transform(
            X_train_text
        )
    )

    X_test = (
        vectorizer.transform(
            X_test_text
        )
    )

    vectorization_time = (
        time.time() - start_time
    )

    vocabulary_size = len(
        vectorizer.vocabulary_
    )

    print(
        f"✓ TF-IDF completed in "
        f"{vectorization_time:.2f} seconds"
    )

    print(
        f"\nTraining features: "
        f"{X_train.shape}"
    )

    print(
        f"Test features:     "
        f"{X_test.shape}"
    )

    print(
        f"Vocabulary size:   "
        f"{vocabulary_size}"
    )

    # --------------------------------------------------------
    # 7. START FINAL MLFLOW RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="final-linear-svm"
    ):

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            MODEL_TYPE,
        )

        mlflow.log_param(
            "C",
            BEST_C,
        )

        mlflow.log_param(
            "max_iter",
            MAX_ITER,
        )

        mlflow.log_param(
            "training_samples",
            len(X_train_text),
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

        mlflow.log_param(
            "selection_method",
            "validation_macro_f1",
        )

        mlflow.log_param(
            "selected_from",
            "svm_hyperparameter_tuning",
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # ----------------------------------------------------
        # Train final model
        # ----------------------------------------------------

        print(
            "\nTraining final Linear SVM..."
        )

        model = LinearSVC(
            C=BEST_C,
            max_iter=MAX_ITER,
            random_state=42,
        )

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.time() - start_time
        )

        print(
            f"✓ Final model trained in "
            f"{training_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Log training time
        # ----------------------------------------------------

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        mlflow.log_metric(
            "vectorization_time_seconds",
            vectorization_time,
        )

        # ----------------------------------------------------
        # FINAL TEST EVALUATION
        # ----------------------------------------------------

        print(
            "\nEvaluating final model "
            "on untouched test set..."
        )

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "test_accuracy",
            test_metrics[
                "accuracy"
            ],
        )

        mlflow.log_metric(
            "test_weighted_precision",
            test_metrics[
                "weighted_precision"
            ],
        )

        mlflow.log_metric(
            "test_weighted_recall",
            test_metrics[
                "weighted_recall"
            ],
        )

        mlflow.log_metric(
            "test_weighted_f1",
            test_metrics[
                "weighted_f1"
            ],
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

        mlflow.log_metric(
            "test_macro_f1",
            test_metrics[
                "macro_f1"
            ],
        )

        mlflow.log_metric(
            "test_inference_time_seconds",
            test_metrics[
                "inference_time"
            ],
        )

        # ----------------------------------------------------
        # Display final results
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "          FINAL TEST RESULTS"
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
        # SAVE MODEL LOCALLY
        # ----------------------------------------------------

        model_path = (
            MODEL_DIR /
            "final_model.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"\n✓ Final model saved to: "
            f"{model_path}"
        )

        # ----------------------------------------------------
        # SAVE VECTORIZER
        # ----------------------------------------------------

        vectorizer_path = (
            MODEL_DIR /
            "final_tfidf_vectorizer.joblib"
        )

        joblib.dump(
            vectorizer,
            vectorizer_path,
        )

        print(
            f"✓ Final vectorizer saved to: "
            f"{vectorizer_path}"
        )

        # ----------------------------------------------------
        # LOG MODEL TO MLFLOW
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="final_linear_svm_model",
        )

        # ----------------------------------------------------
        # LOG VECTORIZER TO MLFLOW
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(vectorizer_path),
            artifact_path="vectorizer",
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
            "\n========================================"
        )

        print(
            "✓ FINAL MODEL PIPELINE COMPLETED"
        )

        print(
            "========================================\n"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
