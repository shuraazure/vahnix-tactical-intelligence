# NTRO Thermal Detection

This project contains a Python + Streamlit dashboard for thermal detection and analysis workflows, including classification, clustering, explainability, reporting, and data ingestion modules.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./startup.sh
```

## Project Structure

- `src/classification` — model training and risk scoring
- `src/clustering` — H3 and DBSCAN-based spatial clustering
- `src/data_pipeline` — ingestion and synthetic data generation
- `src/features` — feature extraction
- `src/explainability` — SHAP and NLP explainability
- `src/reporting` — exports and reporting
- `src/dashboard` — Streamlit application
