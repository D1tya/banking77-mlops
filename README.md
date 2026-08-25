# Banking77 MLOps

An end-to-end MLOps pipeline for classifying customer banking queries into one of 77 intents using TF-IDF and LinearSVC.

The project focuses on taking a machine learning model beyond experimentation and turning it into a reproducible, tested, containerized, monitored API.

---

## Project Overview

Banking applications receive large numbers of natural-language customer requests such as:

- "My card hasn't arrived yet"
- "I need to exchange currencies"
- "Why hasn't my transfer arrived?"
- "How can I get a physical card?"

This project builds an intent-classification system that maps each message to one of the 77 Banking77 intent categories.

The project follows an MLOps-oriented workflow:

```text
                    Banking77 Dataset
                           │
                           ▼
                  Data Validation
                           │
                           ▼
                 Train / Validation
                           │
                           ▼
                    TF-IDF
                           │
                           ▼
                    LinearSVC
                           │
                           ▼
               Hyperparameter Tuning
                           │
                           ▼
                    Final Model
                           │
                           ▼
                       MLflow
                           │
                           ▼
                 Saved Model Artifacts
                           │
                           ▼
                      FastAPI
                           │
                           ▼
                       Docker
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
             Prometheus           CI/CD
                 │
                 ▼
              Grafana
