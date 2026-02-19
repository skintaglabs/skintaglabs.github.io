# SkinTag

AI-powered skin lesion triage. SigLIP fine-tune (Python) + React webapp.

## Commands

- `make venv` — create venv + install deps
- `make app` — start inference API server (port 8000)
- `make preview` — start React dev server
- `make train` — train logistic regression on HAM10000
- `make train-all` — train all 3 models (baseline, logistic, deep)
- `make evaluate` — fairness evaluation on test set
- `make data` — download HAM10000 from Kaggle
- `make pipeline` — full pipeline: data → embed → train → eval

## Architecture

`backend/` — Python inference API
`webapp-react/` — React frontend
`src/` — model training and evaluation
`models/` — saved checkpoints
`configs/` — training hyperparameters
`scripts/` — data prep and utilities
