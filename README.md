# 🌫️ Pearls AQI Predictor (10Pearls Project)

An end-to-end **Machine Learning + Web Dashboard system** for predicting Air Quality Index (AQI) for the next 3 days using historical pollution data, engineered features, weather API integration, and model explainability (SHAP).

---

## 🚀 Project Overview

This project predicts AQI using:
- Time-series engineered features
- XGBoost regression model
- Weather API integration (OpenWeatherMap)
- Interactive Streamlit dashboard
- Model explainability using SHAP

It is designed as a **real-world ML deployment system** with both prediction and interpretability.

---

## 📊 Features

### 🔹 Machine Learning Model
- XGBoost Regressor
- Log-transformed AQI prediction
- Feature engineering:
  - Lag features (1, 3, 7 days)
  - Rolling statistics (mean, std)
  - Differencing trends
  - Time features (day, month, weekday)

### 🔹 Weather API Integration
- Live weather data from OpenWeatherMap API
- Temperature, humidity, wind speed
- City-based input

### 🔹 Explainable AI
- SHAP values for feature importance
- Visual interpretation of model decisions

### 🔹 Forecasting
- 3-day AQI prediction
- Trend visualization using Plotly

### 🔹 Web App
- Built with Streamlit
- Interactive sidebar inputs
- Real-time predictions

---

## 🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Plotly
- OpenWeatherMap API
- Joblib

---

## 📁 Project Structure


AQI Project/
│
├── app.py # Streamlit web application
├── aqi_model.pkl # Trained ML model
├── city_day.csv # Dataset (if included)
├── requirements.txt # Dependencies
└── README.md # Documentation


---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/aqi-predictor.git
cd aqi-predictor
2. Install Dependencies
python -m pip install -r requirements.txt
3. Run Streamlit App
streamlit run app.py
🔑 API Key Setup (Required)

This project uses OpenWeatherMap API.

Steps:
Go to https://openweathermap.org/api
Sign up and generate API key
Open app.py
Replace:
API_KEY = "YOUR_API_KEY"
📈 Model Performance
MAE: ~8–10 (varies by dataset split)
RMSE: ~12–13
R² Score: ~0.97+
📊 Explainability (SHAP)

The model uses SHAP to explain predictions:

Shows how each feature impacts AQI prediction
Highlights lag features as most important
Provides visual bar chart explanation
🌍 Key Insights
Past AQI values (lag features) are the strongest predictors
Seasonal and weekly patterns affect pollution levels
Weather conditions slightly influence AQI trends
⚠️ Notes
Model is trained on historical city-level AQI dataset
API-based weather features enhance real-time prediction
Ensure correct feature order during prediction
👨‍💻 Author

Atifa Nawaz
Bahria University (Karachi Campus)
Data Science Project – 10Pearls Shine Program