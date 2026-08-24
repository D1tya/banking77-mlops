import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.inference.predict import predict
from src.monitoring.metrics import (
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
    PREDICTION_SUCCESS,
)


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
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Banking77 Intent Classification API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "LinearSVC",
        "version": "1.0.0",
    }


# ============================================================
# PROMETHEUS METRICS
# ============================================================

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_intent(
    request: PredictionRequest,
):
    PREDICTION_REQUESTS.inc()

    start_time = time.perf_counter()

    try:
        result = predict(request.text)

        PREDICTION_SUCCESS.inc()

        return result

    except ValueError as error:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except TypeError as error:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except Exception as error:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )

    finally:

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        PREDICTION_LATENCY.observe(
            elapsed_time
        )
