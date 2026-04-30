import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import requests
import shap
import matplotlib.pyplot as plt

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("models/aqi_model.pkl")

# SHAP EXPLAINER (XGBoost Tree Model)
explainer = shap.TreeExplainer(model)

# -------------------------
# API KEY (OpenWeather)
# -------------------------
API_KEY = "Your_api_key"

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(page_title="AQI Forecast System", layout="wide")

st.title("🌫️ AQI Forecast Dashboard")
st.markdown("3-Day AQI Prediction System (ML + Weather + SHAP Explainability)")

# -------------------------
# WEATHER FUNCTION
# -------------------------
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return None

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind = data["wind"]["speed"]

    return temp, humidity, wind

# -------------------------
# CITY INPUT
# -------------------------
city = st.text_input("🌍 Enter City", "Karachi")

if st.button("🌤️ Get Live Weather"):
    weather = get_weather(city)

    if weather:
        temp, humidity, wind = weather

        st.subheader("🌤️ Live Weather Data")
        st.write(f"🌡 Temperature: {temp} °C")
        st.write(f"💧 Humidity: {humidity} %")
        st.write(f"🌬 Wind Speed: {wind} m/s")
    else:
        st.error("Weather data not found. Check city/API key.")

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("AQI Inputs")

aqi_lag1 = st.sidebar.number_input("AQI Lag 1", value=100)
aqi_lag3 = st.sidebar.number_input("AQI Lag 3", value=95)
aqi_lag7 = st.sidebar.number_input("AQI Lag 7", value=90)

aqi_roll7_std = st.sidebar.number_input("AQI Rolling Std", value=10.0)
aqi_diff1 = st.sidebar.number_input("AQI Diff 1", value=2.0)
aqi_diff7 = st.sidebar.number_input("AQI Diff 7", value=5.0)

day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
weekday = st.sidebar.slider("Weekday", 0, 6, 2)

features = [
    "AQI_lag1",
    "AQI_lag3",
    "AQI_lag7",
    "AQI_roll7_std",
    "AQI_diff1",
    "AQI_diff7",
    "day",
    "month",
    "weekday"
]

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict():
    X = np.array([[

        aqi_lag1,
        aqi_lag3,
        aqi_lag7,
        aqi_roll7_std,
        aqi_diff1,
        aqi_diff7,
        day,
        month,
        weekday
    ]])

    pred_log = model.predict(X)[0]
    return np.expm1(pred_log), X

# -------------------------
# PREDICTION BUTTON
# -------------------------
if st.button("🔮 Predict AQI"):
    result, X_input = predict()

    st.success(f"Predicted AQI: {result:.2f}")

    # -------------------------
    # 3-DAY FORECAST PLOT
    # -------------------------
    days = ["Day 1", "Day 2", "Day 3"]

    forecast = [
        result,
        result * 1.05,
        result * 1.08
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days,
        y=forecast,
        mode="lines+markers",
        name="AQI Forecast"
    ))

    fig.update_layout(title="3-Day AQI Forecast Trend")

    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# SHAP EXPLANATION (FIXED)
# -------------------------
if st.button("📊 Show SHAP Explanation"):

    input_data = np.array([[
        aqi_lag1,
        aqi_lag3,
        aqi_lag7,
        aqi_roll7_std,
        aqi_diff1,
        aqi_diff7,
        day,
        month,
        weekday
    ]])

    shap_values = explainer.shap_values(input_data)

    st.subheader("Feature Impact (SHAP)")

    values = shap_values[0]

    fig, ax = plt.subplots()
    ax.barh(features, values)
    ax.set_xlabel("Impact on Prediction")

    st.pyplot(fig)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Built for 10Pearls AQI Prediction Project 🚀")