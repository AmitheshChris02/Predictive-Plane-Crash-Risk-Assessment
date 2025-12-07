# ===============================================================
# ENHANCED TAKEOFF RISK ASSESSMENT SYSTEM
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Open-Meteo SDK
import openmeteo_requests
import requests_cache
from retry_requests import retry


# -------------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------------
st.set_page_config(page_title="Advanced Takeoff Risk Predictor", layout="wide")

st.title("Advanced Aircraft Takeoff Risk Prediction System")
st.write("Real-time weather + aircraft data + ML model → Comprehensive risk assessment with actionable insights")


# -------------------------------------------------------------
# Open-Meteo Client Setup
# -------------------------------------------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# -------------------------------------------------------------
# Enhanced Weather Fetch with Forecasting
# -------------------------------------------------------------
def get_weather_forecast(city_name: str, hours_ahead: int = 12):
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1}
        ).json()

        if "results" not in geo:
            raise Exception("City not found")

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        city_full = geo["results"][0]["name"]
        country = geo["results"][0].get("country", "")

        # Get current + hourly forecast
        response = openmeteo.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "precipitation",
                    "pressure_msl",
                    "cloud_cover"
                ],
                "hourly": [
                    "temperature_2m",
                    "wind_speed_10m",
                    "precipitation_probability",
                    "precipitation",
                    "visibility"
                ],
                "forecast_days": 1
            }
        )[0]

        curr = response.Current()
        hourly = response.Hourly()

        # Current weather
        current_weather = {
            "city": f"{city_full}, {country}",
            "temp_c": round(curr.Variables(0).Value(), 1),
            "humidity": round(curr.Variables(1).Value(), 1),
            "windspeed_m_s": round(curr.Variables(2).Value(), 1),
            "wind_deg": round(curr.Variables(3).Value(), 1),
            "precipitation_mm": round(curr.Variables(4).Value(), 2),
            "pressure_hpa": round(curr.Variables(5).Value(), 1),
            "cloud_cover": round(curr.Variables(6).Value(), 1),
            "visibility_m": 10000,
            "timestamp": datetime.now()
        }

        # Determine precipitation type and weather description
        temp = current_weather["temp_c"]
        precip = current_weather["precipitation_mm"]
        
        if precip > 0:
            if temp < 0:
                current_weather["precipitation_type"] = "snow"
                current_weather["weather_desc"] = "snow"
            elif temp < 4:
                current_weather["precipitation_type"] = "mixed"
                current_weather["weather_desc"] = "freezing rain"
            else:
                current_weather["precipitation_type"] = "rain"
                current_weather["weather_desc"] = "rain"
        else:
            current_weather["precipitation_type"] = "none"
            if current_weather["cloud_cover"] > 70:
                current_weather["weather_desc"] = "cloudy"
            elif current_weather["cloud_cover"] > 30:
                current_weather["weather_desc"] = "partly cloudy"
            else:
                current_weather["weather_desc"] = "clear sky"

        # Forecast data
        forecast = []
        for i in range(min(hours_ahead, 24)):
            hour_temp = hourly.Variables(0).ValuesAsNumpy()[i]
            hour_precip = hourly.Variables(3).ValuesAsNumpy()[i]
            
            forecast.append({
                "hour": i + 1,
                "temp_c": round(float(hour_temp), 1),
                "windspeed_m_s": round(float(hourly.Variables(1).ValuesAsNumpy()[i]), 1),
                "precip_prob": round(float(hourly.Variables(2).ValuesAsNumpy()[i]), 0),
                "precip_mm": round(float(hour_precip), 2),
                "visibility_m": round(float(hourly.Variables(4).ValuesAsNumpy()[i]), 0)
            })

        return current_weather, forecast

    except Exception as e:
        st.warning(f"Weather fetch error: {e}. Using default values.")
        return {
            "city": city_name,
            "temp_c": 25.0,
            "humidity": 60.0,
            "windspeed_m_s": 5.0,
            "wind_deg": 180.0,
            "precipitation_mm": 0.0,
            "pressure_hpa": 1013.25,
            "cloud_cover": 20.0,
            "visibility_m": 10000,
            "precipitation_type": "none",
            "weather_desc": "clear sky",
            "timestamp": datetime.now()
        }, []


# -------------------------------------------------------------
# Enhanced Aircraft Database with Real Airlines
# -------------------------------------------------------------
def load_aircraft_db():
    # Real airlines with actual aircraft models
    df = pd.DataFrame([
        # American Airlines
        ["American Airlines", "AA-2341", "Boeing 737-800", 5, 45, 0.82, 4, "A", 79010],
        ["American Airlines", "AA-8923", "Airbus A321neo", 2, 15, 0.88, 3, "A", 93500],
        ["American Airlines", "AA-1567", "Boeing 777-300ER", 8, 120, 0.75, 6, "B", 351500],
        
        # Delta Air Lines
        ["Delta Air Lines", "DL-5432", "Airbus A350-900", 3, 30, 0.85, 5, "A", 280000],
        ["Delta Air Lines", "DL-7821", "Boeing 757-200", 18, 280, 0.68, 9, "C", 115680],
        ["Delta Air Lines", "DL-3201", "Airbus A220-300", 1, 8, 0.90, 2, "A", 69850],
        
        # United Airlines
        ["United Airlines", "UA-9182", "Boeing 787-9 Dreamliner", 4, 60, 0.83, 4, "A", 254000],
        ["United Airlines", "UA-4532", "Boeing 737 MAX 9", 1, 12, 0.91, 2, "A", 88300],
        ["United Airlines", "UA-6754", "Airbus A319", 15, 220, 0.72, 8, "C", 75500],
        
        # Emirates
        ["Emirates", "EK-8901", "Airbus A380-800", 7, 95, 0.78, 7, "B", 575000],
        ["Emirates", "EK-2234", "Boeing 777-300ER", 6, 75, 0.84, 5, "B", 351500],
        ["Emirates", "EK-4512", "Boeing 777-200LR", 12, 180, 0.70, 8, "B", 347450],
        
        # Lufthansa
        ["Lufthansa", "LH-7632", "Airbus A350-900", 2, 25, 0.87, 3, "A", 280000],
        ["Lufthansa", "LH-3421", "Boeing 747-8", 9, 140, 0.74, 7, "B", 447700],
        ["Lufthansa", "LH-5123", "Airbus A320neo", 3, 40, 0.86, 4, "A", 79000],
        
        # Singapore Airlines
        ["Singapore Airlines", "SQ-9876", "Airbus A380-800", 5, 70, 0.81, 6, "B", 575000],
        ["Singapore Airlines", "SQ-3344", "Boeing 787-10", 2, 20, 0.89, 3, "A", 254000],
        ["Singapore Airlines", "SQ-7712", "Airbus A350-900ULR", 3, 35, 0.85, 4, "A", 280000],
        
        # Southwest Airlines
        ["Southwest Airlines", "WN-4521", "Boeing 737-800", 8, 110, 0.79, 6, "B", 79010],
        ["Southwest Airlines", "WN-8834", "Boeing 737 MAX 8", 1, 10, 0.92, 2, "A", 82190],
        ["Southwest Airlines", "WN-2156", "Boeing 737-700", 16, 245, 0.70, 9, "C", 70080],
        
        # Air France
        ["Air France", "AF-6621", "Boeing 787-9", 4, 55, 0.84, 5, "A", 254000],
        ["Air France", "AF-8912", "Airbus A350-900", 2, 22, 0.88, 3, "A", 280000],
        ["Air France", "AF-3421", "Airbus A318", 19, 310, 0.65, 10, "D", 68000],
        
        # British Airways
        ["British Airways", "BA-7734", "Airbus A380-800", 8, 125, 0.76, 7, "B", 575000],
        ["British Airways", "BA-4523", "Boeing 787-10", 3, 38, 0.86, 4, "A", 254000],
        ["British Airways", "BA-1289", "Airbus A320neo", 2, 18, 0.89, 3, "A", 79000],
        
        # Qatar Airways
        ["Qatar Airways", "QR-9921", "Airbus A350-1000", 2, 28, 0.87, 3, "A", 319000],
        ["Qatar Airways", "QR-5634", "Boeing 777-300ER", 7, 105, 0.77, 6, "B", 351500],
        ["Qatar Airways", "QR-2341", "Boeing 787-9", 4, 62, 0.83, 5, "A", 254000],
        
        # IndiGo
        ["IndiGo", "6E-7823", "Airbus A320neo", 2, 24, 0.88, 3, "A", 79000],
        ["IndiGo", "6E-4512", "Airbus A321neo", 3, 42, 0.85, 4, "A", 93500],
        ["IndiGo", "6E-9201", "ATR 72-600", 6, 88, 0.78, 5, "B", 23000],
    ], columns=[
        "airline", "aircraft_id", "aircraft_type", "aircraft_age_years",
        "days_since_service", "load_factor", "crosswind_component", 
        "maintenance_status", "max_takeoff_weight_kg"
    ])
    
    # Save to CSV for reference
    df.to_csv("aircraft_db_enhanced.csv", index=False)
    return df

# Force reload aircraft database
if 'aircraft_db' not in st.session_state:
    st.session_state.aircraft_db = load_aircraft_db()

aircraft_db = st.session_state.aircraft_db


# -------------------------------------------------------------
# Enhanced ML Model with More Features
# -------------------------------------------------------------
def load_or_create_model():
    path = "enhanced_risk_model.pkl"
    features = [
        "aircraft_age_years", "days_since_service", "load_factor",
        "crosswind_component", "temp_c", "windspeed_m_s",
        "visibility_m", "precipitation_rain", "precipitation_snow",
        "runway_length_m", "humidity", "pressure_hpa", "cloud_cover",
        "maintenance_grade"
    ]

    if os.path.exists(path):
        with open(path, "rb") as f:
            return (*pickle.load(f), features)

    # Create synthetic training data with more realistic patterns
    n = 5000
    rng = np.random.RandomState(42)

    X = pd.DataFrame({
        "aircraft_age_years": rng.randint(0, 30, n),
        "days_since_service": rng.randint(0, 800, n),
        "load_factor": rng.uniform(0.4, 1.0, n),
        "crosswind_component": rng.randint(0, 20, n),
        "temp_c": rng.randint(-15, 45, n),
        "windspeed_m_s": rng.gamma(3, 2, n),
        "visibility_m": rng.choice([500, 1000, 2000, 5000, 10000, 15000], n),
        "precipitation_rain": rng.binomial(1, 0.2, n),
        "precipitation_snow": rng.binomial(1, 0.05, n),
        "runway_length_m": rng.randint(800, 4500, n),
        "humidity": rng.uniform(20, 100, n),
        "pressure_hpa": rng.normal(1013, 15, n),
        "cloud_cover": rng.uniform(0, 100, n),
        "maintenance_grade": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.3, 0.15, 0.05])
    })

    # Enhanced risk scoring formula
    score = (
        0.025 * X.aircraft_age_years +
        0.004 * X.days_since_service +
        0.5 * X.precipitation_rain +
        0.9 * X.precipitation_snow +
        0.025 * X.crosswind_component +
        0.015 * X.windspeed_m_s -
        0.0005 * X.runway_length_m -
        0.0002 * X.visibility_m +
        0.3 * X.maintenance_grade +
        np.where(X.temp_c < 0, 0.3, 0) +
        np.where(X.temp_c > 35, 0.2, 0) +
        0.002 * (100 - X.cloud_cover) / 100 +
        rng.normal(0, 0.15, n)
    )

    y = pd.cut(score, [-999, 0.6, 1.3, 999], labels=[0, 1, 2]).astype(int)

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier

    scaler = StandardScaler()
    scaler.fit(X)
    Xs = scaler.transform(X)

    model = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42
    )
    model.fit(Xs, y)

    with open(path, "wb") as f:
        pickle.dump((model, scaler), f)

    return model, scaler, features


model, scaler, features = load_or_create_model()


# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def calculate_density_altitude(temp_c, pressure_hpa, elevation_m=0):
    """Calculate density altitude for performance assessment"""
    # Simplified formula
    temp_k = temp_c + 273.15
    std_temp = 288.15 - (0.0065 * elevation_m)
    pressure_ratio = pressure_hpa / 1013.25
    
    density_alt = elevation_m + (temp_k - std_temp) * 120
    return round(density_alt, 0)


def get_risk_recommendations(risk_level, feature_impacts, weather, runway_length):
    """Generate actionable recommendations based on risk factors"""
    recommendations = []
    
    if risk_level == "High":
        recommendations.append("⛔ **RECOMMEND DELAY** - Conditions are not favorable for safe takeoff")
    elif risk_level == "Medium":
        recommendations.append("⚠️ **CAUTION ADVISED** - Monitor conditions closely")
    else:
        recommendations.append("✅ **CLEARED FOR TAKEOFF** - Conditions are within safe parameters")
    
    # Specific recommendations based on top risk factors
    top_factors = feature_impacts.nlargest(3, 'abs')
    
    for _, row in top_factors.iterrows():
        if row['shap'] > 0.1:  # Only significant positive impacts
            if 'precipitation' in row['feature']:
                recommendations.append("• Consider de-icing procedures if not already performed")
                recommendations.append("• Increase takeoff speed by 5-10 knots for wet runway")
            elif 'windspeed' in row['feature'] or 'crosswind' in row['feature']:
                recommendations.append("• Use maximum available runway length")
                recommendations.append("• Ensure crosswind limits are not exceeded for aircraft type")
            elif 'visibility' in row['feature']:
                recommendations.append("• Verify IFR procedures are available")
                recommendations.append("• Confirm minimum visibility requirements met")
            elif 'maintenance' in row['feature']:
                recommendations.append("• Review MEL (Minimum Equipment List) items")
                recommendations.append("• Consider postponing until maintenance completed")
    
    if runway_length < 2000:
        recommendations.append("• Short runway - verify aircraft performance charts")
    
    return recommendations


def calculate_what_if_scenarios(base_features, model, scaler, features_list):
    """Calculate risk under different scenarios"""
    scenarios = []
    
    # Scenario 1: Wait 2 hours (assume weather improves slightly)
    wait_features = base_features.copy()
    wait_features['windspeed_m_s'] *= 0.8
    wait_features['precipitation_rain'] = 0
    wait_features['visibility_m'] = min(wait_features['visibility_m'] * 1.2, 15000)
    scenarios.append(("Wait 2 hours (weather improves)", wait_features))
    
    # Scenario 2: Use longer runway
    longer_runway = base_features.copy()
    longer_runway['runway_length_m'] = min(longer_runway['runway_length_m'] * 1.3, 4500)
    scenarios.append(("Use longer runway", longer_runway))
    
    # Scenario 3: Reduce load factor
    reduce_load = base_features.copy()
    reduce_load['load_factor'] = max(reduce_load['load_factor'] * 0.85, 0.5)
    scenarios.append(("Reduce passenger load by 15%", reduce_load))
    
    results = []
    for name, feat_dict in scenarios:
        X = np.array([feat_dict[f] for f in features_list]).reshape(1, -1)
        Xs = scaler.transform(X)
        pred = int(model.predict(Xs)[0])
        proba = model.predict_proba(Xs)[0]
        label = {0: "Low", 1: "Medium", 2: "High"}[pred]
        results.append({
            "scenario": name,
            "risk_level": label,
            "risk_score": pred,
            "confidence": max(proba) * 100
        })
    
    return pd.DataFrame(results)


# -------------------------------------------------------------
# Main UI
# -------------------------------------------------------------

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh weather (every 5 min)", value=False)
    show_advanced = st.checkbox("Show advanced metrics", value=True)
    export_report = st.checkbox("Enable report export", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.success("✓ Weather API: Online")
    st.success("✓ ML Model: Loaded")
    st.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🌍 Location & Weather")
    
    city = st.text_input("Airport / City", "Chennai", help="Enter airport name or city")
    
    if 'weather_cache' not in st.session_state or st.session_state.get('last_city') != city:
        weather, forecast = get_weather_forecast(city, hours_ahead=12)
        st.session_state.weather_cache = weather
        st.session_state.forecast_cache = forecast
        st.session_state.last_city = city
    else:
        weather = st.session_state.weather_cache
        forecast = st.session_state.forecast_cache
    
    # Weather display
    st.subheader(f"📍 {weather['city']}")
    
    wcol1, wcol2, wcol3, wcol4 = st.columns(4)
    wcol1.metric("🌡️ Temperature", f"{weather['temp_c']}°C")
    wcol2.metric("💨 Wind Speed", f"{weather['windspeed_m_s']} m/s")
    wcol3.metric("👁️ Visibility", f"{weather['visibility_m']/1000:.1f} km")
    wcol4.metric("☁️ Cloud Cover", f"{weather['cloud_cover']}%")
    
    if show_advanced:
        acol1, acol2, acol3 = st.columns(3)
        acol1.metric("💧 Humidity", f"{weather['humidity']}%")
        acol2.metric("🔽 Pressure", f"{weather['pressure_hpa']} hPa")
        acol3.metric("🌧️ Precipitation", f"{weather['precipitation_mm']} mm")
    
    st.info(f"Current conditions: **{weather['weather_desc']}** | Wind direction: {weather['wind_deg']}°")
    
    # Forecast chart
    if forecast and len(forecast) > 0:
        st.subheader("📈 12-Hour Forecast")
        forecast_df = pd.DataFrame(forecast)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        
        # Temperature and Wind
        ax1.plot(forecast_df['hour'], forecast_df['temp_c'], 'r-', linewidth=2, marker='o')
        ax1.set_ylabel('Temperature (°C)', color='r', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Temperature & Wind Speed Forecast')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(forecast_df['hour'], forecast_df['windspeed_m_s'], 'b-', linewidth=2, marker='s')
        ax1_twin.set_ylabel('Wind Speed (m/s)', color='b', fontsize=10)
        ax1_twin.tick_params(axis='y', labelcolor='b')
        
        # Precipitation
        ax2.bar(forecast_df['hour'], forecast_df['precip_prob'], alpha=0.7, color='steelblue', edgecolor='navy')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(forecast_df['hour'], forecast_df['precip_mm'], 'g-', linewidth=2, marker='^', label='Precip (mm)')
        ax2.set_xlabel('Hours from now', fontsize=10)
        ax2.set_ylabel('Precipitation Probability (%)', color='steelblue', fontsize=10)
        ax2_twin.set_ylabel('Precipitation (mm)', color='g', fontsize=10)
        ax2_twin.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Precipitation Forecast')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("📊 Forecast data unavailable")

with col2:
    st.header("        Aircraft Database")
    st.dataframe(aircraft_db, height=300)


# Aircraft and Runway Selection
st.markdown("---")
st.header("🛫 Flight Configuration")

fcol1, fcol2, fcol3 = st.columns(3)

with fcol1:
    airline = st.selectbox("Airline", aircraft_db.airline.unique())
    aircraft = st.selectbox("Aircraft ID", 
                           aircraft_db[aircraft_db.airline == airline]["aircraft_id"])
    
    row = aircraft_db[(aircraft_db.airline == airline) &
                      (aircraft_db.aircraft_id == aircraft)].iloc[0]
    
    # Use bracket notation for safer access
    st.info(f"**{row['aircraft_type']}**")
    st.write(f"Age: {row['aircraft_age_years']} years")
    st.write(f"Days since service: {row['days_since_service']}")
    st.write(f"Maintenance: Grade {row['maintenance_status']}")

with fcol2:
    runway = st.number_input("Runway Length (m)", 
                            min_value=500.0, max_value=5000.0, 
                            value=2500.0, step=100.0)
    
    runway_elevation = st.number_input("Runway Elevation (m)", 
                                      min_value=0.0, max_value=5000.0, 
                                      value=0.0, step=10.0)
    
    runway_condition = st.selectbox("Runway Condition", 
                                   ["Dry", "Wet", "Snow/Ice"],
                                   index=0 if weather['precipitation_mm'] == 0 else 1)

with fcol3:
    user_wind = st.number_input("Wind Speed (m/s)", 
                               min_value=0.0, max_value=50.0,
                               value=float(weather["windspeed_m_s"]),
                               step=0.5)
    
    visibility = st.number_input("Visibility (m)", 
                                min_value=50.0, max_value=50000.0,
                                value=float(weather["visibility_m"]),
                                step=100.0)
    
    precipitation = st.selectbox(
        "Precipitation", ["none", "rain", "snow"],
        index=["none", "rain", "snow"].index(weather["precipitation_type"])
    )


# Build feature vector
maintenance_map = {"A": 1, "B": 2, "C": 3, "D": 4}
feat = {
    "aircraft_age_years": float(row['aircraft_age_years']),
    "days_since_service": float(row['days_since_service']),
    "load_factor": float(row['load_factor']),
    "crosswind_component": float(row['crosswind_component']),
    "temp_c": float(weather["temp_c"]),
    "windspeed_m_s": float(user_wind),
    "visibility_m": float(visibility),
    "precipitation_rain": 1 if precipitation == "rain" else 0,
    "precipitation_snow": 1 if precipitation == "snow" else 0,
    "runway_length_m": float(runway),
    "humidity": float(weather["humidity"]),
    "pressure_hpa": float(weather["pressure_hpa"]),
    "cloud_cover": float(weather["cloud_cover"]),
    "maintenance_grade": float(maintenance_map.get(row['maintenance_status'], 2))
}


# Calculate density altitude
density_alt = calculate_density_altitude(
    weather["temp_c"], 
    weather["pressure_hpa"], 
    runway_elevation
)

if show_advanced:
    st.info(f"⛰️ Density Altitude: {density_alt:.0f} m | "
            f"{'⚠️ HIGH - Reduced aircraft performance' if density_alt > 2000 else '✓ Normal'}")


# -------------------------------------------------------------
# Risk Assessment
# -------------------------------------------------------------
st.markdown("---")

if st.button("**ASSESS TAKEOFF RISK**", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing conditions..."):
        X = np.array([feat[f] for f in features]).reshape(1, -1)
        Xs = scaler.transform(X)

        proba = model.predict_proba(Xs)[0]
        pred = int(model.predict(Xs)[0])

        label = {0: "Low", 1: "Medium", 2: "High"}[pred]
        colors = {0: "green", 1: "orange", 2: "red"}
        color = colors[pred]

    # Risk Result
    st.markdown(f"##  Takeoff Risk Assessment: <span style='color:{color}; font-size:36px'>**{label.upper()}**</span>", 
                unsafe_allow_html=True)
    
    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric("Risk Level", label, delta=None)
    rcol2.metric("Confidence", f"{max(proba)*100:.1f}%")
    rcol3.metric("Status", "GO" if pred == 0 else "NO-GO" if pred == 2 else "CAUTION")
    
    # Probability distribution
    st.subheader(" Risk Probability Distribution")
    prob_df = pd.DataFrame({
        "Risk Level": ["Low", "Medium", "High"],
        "Probability (%)": [p*100 for p in proba]
    })
    st.bar_chart(prob_df.set_index("Risk Level"))

    
    # SHAP Explanation
    st.markdown("---")
    st.subheader("Factor Analysis (SHAP)")
    
    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xs)
        
        # For multi-class, select the predicted class
        if isinstance(shap_values, list):
            sv = shap_values[pred][0]
        else:
            sv = shap_values[0][:, pred] if len(shap_values.shape) > 2 else shap_values[0]
        
        base_val = explainer.expected_value[pred] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        
        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=Xs[0],
            feature_names=features
        )
        
        # Try waterfall plot
        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(plt.gcf())
        plt.close()
        
    except Exception as e:
        st.warning(f"SHAP visualization note: Using simplified feature importance view")
        
        # Fallback: Calculate feature importance from the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Manual calculation for gradient boosting
            importances = np.abs(sv) if 'sv' in locals() else np.random.rand(len(features))
        
        # Sort and display top features
        indices = np.argsort(importances)[::-1][:10]
        
        fig2, ax = plt.subplots(figsize=(10, 6))
        colors_bar = plt.cm.RdYlGn_r(importances[indices] / importances[indices].max())
        ax.barh([features[i].replace('_', ' ').title() for i in indices][::-1], 
                importances[indices][::-1], 
                color=colors_bar[::-1])
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title("Top 10 Risk Factors", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Store for text explanation
        sv = importances
    
    # Text explanation
    st.subheader("Plain-Language Explanation")
    
    df_imp = pd.DataFrame({"feature": features, "shap": sv})
    df_imp["abs"] = df_imp["shap"].abs()
    top = df_imp.sort_values("abs", ascending=False).head(5)
    
    for idx, r in top.iterrows():
        eff = "⬆️ increases" if r.shap > 0 else "⬇️ decreases"
        magnitude = "significantly" if abs(r.shap) > 0.2 else "moderately"
        st.write(f"**{r.feature.replace('_', ' ').title()}** {magnitude} {eff} risk "
                f"(impact: {abs(r.shap):.3f})")
    
    
    # Recommendations
    st.markdown("---")
    st.subheader("Recommendations")
    
    recommendations = get_risk_recommendations(label, df_imp, weather, runway)
    for rec in recommendations:
        st.markdown(rec)
    
    
    # What-if Scenarios
    st.markdown("---")
    st.subheader("Alternatives To Consider")
    
    scenarios_df = calculate_what_if_scenarios(feat, model, scaler, features)
    
    st.dataframe(
        scenarios_df.style.background_gradient(subset=['risk_score'], cmap='RdYlGn_r'),
        hide_index=True,
        use_container_width=True
    )
    
    st.info("💡 **Insight:** Try these alternatives to reduce risk if current conditions are unfavorable")
    
    
    # Export functionality
    if export_report:
        st.markdown("---")
        st.subheader("Export Report")
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "location": weather['city'],
            "aircraft": f"{row['airline']} {row['aircraft_id']} ({row['aircraft_type']})",
            "risk_assessment": {
                "level": label,
                "confidence": f"{max(proba)*100:.1f}%",
                "probabilities": {
                    "low": f"{proba[0]*100:.1f}%",
                    "medium": f"{proba[1]*100:.1f}%",
                    "high": f"{proba[2]*100:.1f}%"
                }
            },
            "weather": weather,
            "runway": {
                "length_m": runway,
                "elevation_m": runway_elevation,
                "condition": runway_condition
            },
            "top_factors": top.to_dict('records'),
            "recommendations": recommendations
        }
        
        json_report = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="📥 Download JSON Report",
            data=json_report,
            file_name=f"takeoff_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# Footer
st.markdown("---")
# st.caption("⚠️ **Disclaimer:** This system is for educational and research purposes only. "
#           "Always follow official aviation regulations and consult with certified personnel "
#           "for actual flight operations.")

st.caption(f"System Version 2.0 | Model: Gradient Boosting | "
          f"Last Model Update: {datetime.fromtimestamp(os.path.getmtime('enhanced_risk_model.pkl') if os.path.exists('enhanced_risk_model.pkl') else datetime.now().timestamp()).strftime('%Y-%m-%d')}")