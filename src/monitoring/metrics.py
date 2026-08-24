from prometheus_client import Counter, Histogram


# ============================================================
# REQUEST METRICS
# ============================================================

PREDICTION_REQUESTS = Counter(
    "banking77_prediction_requests_total",
    "Total number of prediction requests.",
)

PREDICTION_SUCCESS = Counter(
    "banking77_prediction_success_total",
    "Total number of successful predictions.",
)

PREDICTION_ERRORS = Counter(
    "banking77_prediction_errors_total",
    "Total number of failed predictions.",
)


# ============================================================
# LATENCY METRICS
# ============================================================

PREDICTION_LATENCY = Histogram(
    "banking77_prediction_latency_seconds",
    "Prediction request latency in seconds.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
)
