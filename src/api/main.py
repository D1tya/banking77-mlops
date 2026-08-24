from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predict import predict


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Banking77 Intent Classification API",
    description=(
        "Production-style REST API for Banking77 "
        "intent classification using TF-IDF and LinearSVC."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Customer message to classify.",
    )


class PredictionItem(BaseModel):
    category: str
    score: float


class PredictionResponse(BaseModel):
    category: str
    score: float
    top_predictions: List[PredictionItem]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "LinearSVC",
        "version": "1.0.0",
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_intent(
    request: PredictionRequest,
):

    try:

        result = predict(
            request.text
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except TypeError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Banking77 Intent Classification API"
        ),
        "docs": "/docs",
        "health": "/health",
    }

