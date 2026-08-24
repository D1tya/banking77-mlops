from pathlib import Path

import mlflow


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
    """
    Configure MLflow to use a SQLite backend
    and create/select the Banking77 experiment.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    experiment = (
        mlflow.get_experiment_by_name(
            EXPERIMENT_NAME
        )
    )

    if experiment is None:

        experiment_id = (
            mlflow.create_experiment(
                EXPERIMENT_NAME
            )
        )

        print(
            f"✓ Created MLflow experiment: "
            f"{EXPERIMENT_NAME}"
        )

    else:

        experiment_id = (
            experiment.experiment_id
        )

        print(
            f"✓ Using existing MLflow experiment: "
            f"{EXPERIMENT_NAME}"
        )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    return experiment_id


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    experiment_id = setup_mlflow()

    print(
        f"\nMLflow experiment ID: "
        f"{experiment_id}"
    )

    print(
        f"MLflow tracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )
