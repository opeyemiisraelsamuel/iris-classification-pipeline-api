# Iris Model Pipeline + REST API + Monitoring Analysis

## Files
- `train_pipeline.py` — data preprocessing → training → evaluation. Run this first.
- `model.joblib` — the trained pipeline (produced by `train_pipeline.py`)
- `baseline_stats.json` — training-data feature statistics, used as a drift baseline
- `evaluation_report.txt` — accuracy/precision/recall/F1 on the held-out test set
- `api.py` — FastAPI REST API that serves predictions from `model.joblib`
- `monitoring_analysis.md` — written analysis of how to monitor this model for performance degradation in production
- `requirements.txt` — dependencies

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (creates `model.joblib`, `baseline_stats.json`, `evaluation_report.txt`):
   ```
   python train_pipeline.py
   ```

3. Start the API:
   ```
   uvicorn api:app --reload
   ```

4. Test it:
   ```
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
   ```
   Or open http://127.0.0.1:8000/docs for interactive Swagger docs.

## Notes on choices made
- **Dataset**: Iris — small, clean, public, and lets the pipeline/API/monitoring
  logic stand out rather than time being spent on data cleaning.
- **Model**: Logistic Regression — simple, fast, and easy to explain; the
  assignment is testing pipeline/deployment/monitoring thinking, not model
  sophistication.
- **Framework**: FastAPI — minimal boilerplate, built-in request validation
  via Pydantic, and automatic interactive docs at `/docs`.
- Preprocessing and the model are bundled into a single scikit-learn
  `Pipeline`, so the exact transformations used in training are guaranteed
  to be applied identically at inference time.
