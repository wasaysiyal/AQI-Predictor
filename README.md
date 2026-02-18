# 🌫️ AQI Forecasting System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature%20Store-orange)
![MLOps](https://img.shields.io/badge/MLOps-CI/CD-green)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)


An end-to-end production-grade machine learning pipeline for multi-day Air Quality Index (AQI) forecasting using Hopsworks Feature Store, XGBoost, GitHub Actions, and Streamlit Cloud.

⚠️ Ownership Notice:
This project is independently designed and developed by Wasay Siyal.
Reproduction, redistribution, or reuse of any part of this repository without explicit permission is prohibited.

Python • Hopsworks • GitHub Actions • Streamlit • MLOps • XGBoost

🧭 Project Overview

The AQI Forecasting System is a production-oriented machine learning platform that predicts 1-day, 2-day, and 3-day AQI values using historical environmental data.

Unlike experimental notebooks, this system demonstrates:

Feature Store integration

Versioned data pipelines

Model performance tracking

Batch inference architecture

Cloud deployment

CI/CD automation

Production-safe app design

The entire pipeline — from feature engineering to cloud dashboard visualization — follows modern MLOps principles.

---

## 🧠 Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn, XGBoost |
| Feature Store | Hopsworks |
| Data Processing | Pandas, NumPy |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |
| Storage | Hopsworks Feature Store |


---

## 🔄 MLOps Workflow

1. Daily data ingestion pipeline
2. Feature store synchronization
3. Automated model retraining
4. Batch inference execution
5. Dashboard auto-refresh deployment
6. CI/CD via GitHub Actions


⚙️ Key Features

✅ Versioned Feature Groups in Hopsworks
✅ Multi-Horizon AQI Forecasting (1–3 days)
✅ Model performance tracking (MAE / RMSE)
✅ Batch inference pipeline
✅ Production-safe Streamlit architecture (no auto inference on load)
✅ Live historical + forecast visualization
✅ AQI color-coded UI classification
✅ Secure API key handling via environment secrets
✅ CI/CD integration via GitHub Actions


🏗️ System Architecture
Open-Meteo API
        ↓
Historical AQI Data
        ↓
Feature Engineering (Lag, Rolling Statistics)
        ↓
Hopsworks Feature Store (daily_aqi_features_v2)
        ↓
Model Training (XGBoost Regression)
        ↓
Batch Inference (Manual Trigger)
        ↓
Predictions Feature Group (aqi_predictions_v2)
        ↓
Streamlit Dashboard (Cloud Deployment)


## 📁 Project Structure

aqi-predictor/
│
├── .github/
│ └── workflows/
│ ├── feature_pipeline.yml
│ └── training_pipeline.yml
│
├── app/
│ └── app.py # Streamlit dashboard
│
├── src/
│ ├── batch_inference.py # Batch prediction logic
│ ├── feature_pipeline.py # Feature engineering pipeline
│ ├── hopsworks_client.py # Hopsworks authentication
│ └── train_model.py # Model training script
│
├── notebooks/
│ ├── 01_eda.ipynb
│ └── model_experiments.ipynb
│
├── models/
│ └── best_model.pkl
│
├── requirements.txt
├── .env (excluded from git)
└── README.md


📊 Data & Feature Engineering
Feature Group: daily_aqi_features_v2

Contains:

Historical AQI

Pollutant concentrations

Lag features

Rolling means

Time-based features

Engineered predictive signals

Feature Group: aqi_predictions_v2

Stores:

Predicted AQI

Prediction horizon (1, 2, 3 days)

Model name & version

Source feature timestamp

Inference timestamp

All data is versioned and reproducible.

🧮 Modeling Strategy
Models Evaluated

Linear Regression

Random Forest

XGBoost

Final Selected Model

XGBoost Regressor

Chosen due to:

Strong nonlinear modeling capability

Superior generalization

Stable multi-horizon forecasting performance

Evaluation Metrics
Metric	Purpose
MAE	Average prediction error
RMSE	Penalizes larger deviations

Model performance is displayed directly on the dashboard.

🔄 Batch Inference Design

Inference does NOT auto-run on app startup.

Instead:

Triggered manually via dashboard button

Uses latest feature snapshot

Stores results in Feature Store

Updates dashboard dynamically

This prevents:

Unnecessary API calls

Startup failures

Feature Store overload

Production-safe design principle applied.

📈 Streamlit Dashboard Features
Forecast Cards

3-day AQI forecast

Color-coded AQI classification

Clean KPI card UI

Horizon labeling

Live Chart

14-day historical AQI

Forecast overlay

Interactive time-series visualization

Model Transparency

Model name displayed

Performance metrics shown

Inference timestamp included

AQI Classification Guide
AQI Range	Category	Color
0–50	Good	🟢
51–100	Moderate	🟡
101–150	Unhealthy (Sensitive)	🟠
151–200	Unhealthy	🔴
201–300	Very Unhealthy	🟣
301–500	Hazardous	⚫
🔄 CI/CD Automation

GitHub Actions pipelines:

Workflow	Purpose
feature_pipeline.yml	Updates Feature Store
training_pipeline.yml	Retrains model
Deployment Sync	Push-to-deploy on Streamlit

Ensures:

Model freshness

Data consistency

Automated reproducibility

🧠 Production & MLOps Highlights

✔ Feature Store versioning
✔ API secret management
✔ Environment isolation
✔ Cloud deployment
✔ Modular architecture
✔ Clear separation of training & inference
✔ No hard-coded credentials
✔ Dashboard stability design

📈 Results Snapshot
Model	RMSE	MAE	Notes
Linear Regression	Higher	Higher	Baseline
Random Forest	Improved	Moderate	Good stability
XGBoost	Lowest	Lowest	✅ Selected
🧰 Setup Guide
1️⃣ Clone Repository
git clone https://github.com/yourusername/aqi-predictor.git
cd aqi-predictor

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Environment

Create .env:

HOPSWORKS_API_KEY=your_api_key_here

5️⃣ Run App Locally
streamlit run app/app.py

🚀 Future Enhancements

🧠 SHAP interpretability
📦 Docker containerization
📡 Real-time inference endpoint
🌆 Multi-city forecasting
📱 AQI alert notifications
📊 Model comparison dashboard

👨‍💻 Author

Abdul Wasay 
Software Engineer


📜 License

This project is licensed under the MIT License.