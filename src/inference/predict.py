from pathlib import Path

import joblib


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_model.joblib"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_tfidf_vectorizer.joblib"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: "
            f"{VECTORIZER_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


# ============================================================
# PREDICTION
# ============================================================

def predict(text):

    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "Input text must be a string."
        )

    if not text.strip():
        raise ValueError(
            "Input text cannot be empty."
        )

    model, vectorizer = (
        load_model()
    )

    # --------------------------------------------------------
    # Transform text
    # --------------------------------------------------------

    features = (
        vectorizer.transform(
            [text]
        )
    )

    # --------------------------------------------------------
    # Predict category
    # --------------------------------------------------------

    prediction = model.predict(
        features
    )[0]

    # --------------------------------------------------------
    # Get SVM decision score
    # --------------------------------------------------------

    decision_scores = (
        model.decision_function(
            features
        )
    )

    # LinearSVC returns one score per class
    scores = decision_scores[0]

    predicted_index = list(
        model.classes_
    ).index(
        prediction
    )

    prediction_score = float(
        scores[predicted_index]
    )

    # --------------------------------------------------------
    # Get top predictions
    # --------------------------------------------------------

    ranked_indices = (
        scores.argsort()[::-1]
    )

    top_predictions = []

    for index in ranked_indices[:3]:

        top_predictions.append(
            {
                "category": str(
                    model.classes_[index]
                ),
                "score": float(
                    scores[index]
                ),
            }
        )

    return {
        "category": str(
            prediction
        ),
        "score": prediction_score,
        "top_predictions": (
            top_predictions
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_inputs = [
        "My card hasn't arrived yet",
        "Why was I charged extra?",
        "I need to exchange currencies",
        "My cash withdrawal is still pending",
    ]

    print(
        "\n========================================"
    )

    print(
        "       BANKING77 INFERENCE TEST"
    )

    print(
        "========================================\n"
    )

    for text in test_inputs:

        result = predict(
            text
        )

        print(
            f"Input: {text}"
        )

        print(
            f"Prediction: "
            f"{result['category']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            "Top predictions:"
        )

        for item in result[
            "top_predictions"
        ]:

            print(
                f"  {item['category']}: "
                f"{item['score']:.4f}"
            )

        print(
            "-" * 50
        )
