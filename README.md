# RGM Copilot

**An end-to-end FMCG Revenue Growth Management analytics assistant — demand forecasting, stockout-risk classification, and a GenAI copilot that answers business questions in plain English, grounded in real model output.**

> Built as a portfolio project to mirror the exact skillset modern Data Insights & AI teams need: statistical/ML modeling, applied GenAI (RAG + function calling), and production-style pipeline thinking — applied to a realistic FMCG distribution scenario (regions, channels, SKUs).

---

## Why this project exists

Most junior data science portfolios show one isolated model on a clean Kaggle dataset. RGM Copilot instead chains **three connected systems together**, the way a real analytics team would:

1. A **demand forecasting model** predicts how much of each product will sell
2. A **stockout-risk classifier** consumes that forecast as an input to flag supply risk *before* it happens
3. A **GenAI layer** sits on top of both, letting a non-technical stakeholder ask "why" and get a grounded answer — not a hallucination

That pipeline — forecast → risk → explain — is the actual shape of decision support in Route-to-Market, Revenue Growth Management, and Supply Chain analytics.

**Note on the data:** this project uses a synthetic FMCG distribution dataset, built to mirror realistic patterns (seasonality, promotions, regional variation, supply constraints). No real company data is used.

---

## Architecture

```
┌─────────────────────┐
│   Data Layer         │  Synthetic FMCG dataset: 5 regions × 4 channels ×
│  (Notebook 1)         │  8 SKUs × 2 years daily (116,800 rows)
└──────────┬────────────┘
           │
┌──────────▼────────────┐
│  Demand Forecasting    │  XGBoost, lag/rolling/calendar features,
│  (Notebook 2)          │  time-based train/test split
└──────────┬────────────┘
           │ predicted_demand feeds in as a feature
┌──────────▼────────────┐
│  Stockout Classifier   │  XGBoost, leakage-safe features,
│  (Notebook 3)          │  class-imbalance handling, threshold tuning
└──────────┬────────────┘
           │ model outputs saved as lookup tables
┌──────────▼────────────┐
│   GenAI Copilot        │  Gemini 2.5 Flash + function calling
│  (Notebook 4)          │  (real data) + RAG (grounded business notes)
└────────────────────────┘
```

---

## Key results

### Demand forecasting (Notebook 2)

| Metric | Naive baseline (last week, same weekday) | XGBoost |
|---|---|---|
| MAE | 12.43 | **7.40** |
| RMSE | 23.56 | **14.15** |
| MAPE | 37.9% | **29.6%** |

**~40% lower error than the naive baseline** — evaluated on a genuine held-out future period (final 6 months of a 2-year dataset), not a random split.

### Stockout-risk classification (Notebook 3)

| Metric | Score |
|---|---|
| ROC-AUC | **0.994** |
| PR-AUC | 0.912 |
| Recall (stockouts caught) | 91.4% at default threshold, 95% at tuned threshold |

Built with deliberate attention to **data leakage** — only features genuinely knowable before the day starts (yesterday's closing stock, the demand forecast, sales history) are used. Evaluated with ROC-AUC/PR-AUC/recall rather than accuracy, since stockouts are a rare class (~4.6% of days) where accuracy alone would be misleading.

The model's top predictor was **`predicted_demand`** — the output of the forecasting model from Notebook 2 — confirming the two models genuinely work together as a pipeline rather than as two disconnected exercises.

### GenAI Copilot (Notebook 4)

- **Function calling**: the assistant calls real Python functions (`get_demand_forecast`, `get_stockout_risk`, `get_top_risk_items`) that query the saved model outputs — so numeric answers are computed, not generated
- **RAG**: a small knowledge base of business-context notes, where every note is derived directly from real statistics computed earlier in the project (channel stockout rates, promo uplift, top-risk combinations) — not invented text
- Answers questions like *"What's the demand forecast for Cola Regular 330ml in Greater Cairo?"* and *"Why do certain SKUs keep showing up as high stockout risk in HoReCa?"* with grounded, explainable answers

---

## Tech stack

| Layer | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| ML modeling | Scikit-learn, XGBoost |
| GenAI | Google Gemini (`gemini-2.5-flash`), `google-genai` SDK, function calling, TF-IDF retrieval |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Project structure

```
RGM-Copilot/
├── notebooks/
│   ├── 01_data_generation_eda.ipynb
│   ├── 02_demand_forecasting.ipynb
│   ├── 03_stockout_classification.ipynb
│   └── 04_genai_rag_layer.ipynb
├── data/
│   └── rgm_copilot_dataset.csv
├── models/
│   ├── demand_forecast_model.pkl
│   ├── demand_forecast_features.pkl
│   ├── stockout_risk_model.pkl
│   └── stockout_risk_features.pkl
└── README.md
```

---

## How to run it

1. Clone this repo
2. Create an environment with Python 3.9+ and install dependencies:
   ```
   pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost joblib google-genai
   ```
3. Run the notebooks in order: `01` → `02` → `03` → `04`
4. For Notebook 4, get a free Gemini API key from [Google AI Studio](https://aistudio.google.com) — you'll be prompted to enter it securely when the notebook runs (it is never stored in the notebook file)

---

## Design decisions worth knowing about

A few deliberate choices, explained honestly rather than glossed over:

- **Global model instead of per-series models**: one XGBoost model is trained across all 160 region/channel/SKU combinations at once, rather than fitting 160 separate time series models. This mirrors the winning approach from the M5 Forecasting Competition and is the standard practice for large-scale retail/FMCG forecasting in production.
- **RAG uses TF-IDF, not an embedding API**: this keeps the project comfortably within a free-tier API budget. The retrieval *pattern* (retrieve relevant context, then augment the LLM prompt with it) is identical to a production embedding-based setup — only the retrieval implementation is lighter-weight here.
- **Synthetic data, not real company data**: built with realistic seasonality, promotions, and supply dynamics, and validated with EDA before modeling — but it is synthetic, and that's stated plainly rather than implied otherwise.

---

## Possible next steps

- Wrap the pipeline in a FastAPI service with real endpoints
- Add a Streamlit or Power BI dashboard for non-technical users
- Deploy to Azure (App Service / Functions)
- Swap the TF-IDF retrieval for a proper embedding-based vector store

---

## Author

**Mahmoud Ahmed**
Final-year Business Information Systems student, AASTMT — Data Science & Analytics
[GitHub](https://github.com/mmahmoudahmedd)
