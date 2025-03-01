import streamlit as st
import joblib

# Load trained models
crop_model = joblib.load("crop_recommendation.pkl")
fertilizer_model = joblib.load("fertilizer_recommendation.pkl")

# Load encoders
crop_encoder = joblib.load("crop_encoder.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")

# Streamlit UI
st.title("🌾 Crop & Fertilizer Recommendation System")
st.markdown("#### Designed by Nivetha Durairaj")
# Get user inputs for Crop Recommendation
st.header("Crop Recommendation")
temperature = st.number_input("🌡 Enter Temperature (°C):", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input("💧 Enter Humidity (%):", min_value=0.0, max_value=100.0, step=0.1)
nitrogen = st.number_input("🧪 Enter Nitrogen (N) level:", min_value=0, max_value=100, step=1)
phosphorous = st.number_input("🧪 Enter Phosphorous (P) level:", min_value=0, max_value=100, step=1)
potassium = st.number_input("🧪 Enter Potassium (K) level:", min_value=0, max_value=100, step=1)
ph = st.number_input("🌍 Enter Soil pH level:", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("🌧 Enter Rainfall (mm):", min_value=0.0, max_value=500.0, step=0.1)

# Crop Recommendation Function
def recommend_crop(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall):
    input_features = [[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]]
    prediction = crop_model.predict(input_features)
    return prediction[0]

# Button to get crop prediction
if st.button("🌱 Recommend Crop"):
    recommended_crop = recommend_crop(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall)
    st.success(f"🌾 Recommended Crop: **{recommended_crop}**")

# -------------------------------------------
# Get user inputs for Fertilizer Recommendation
st.header("🧪 Fertilizer Recommendation")
soil_type = st.selectbox("🌱 Select Soil Type:", ["Sandy", "Clayey", "Loamy"])
crop_type = st.selectbox("🌿 Select Crop Type:", ["Wheat",  "Maize", "Cotton", "Sugarcane"])
moisture = st.number_input("💦 Enter Soil Moisture (%):", min_value=0.0, max_value=100.0, step=0.1)

# Convert categorical data using encoders
soil_encoded = soil_encoder.transform([soil_type])[0]
crop_encoded = crop_encoder.transform([crop_type])[0]

# Fertilizer Recommendation Function
def recommend_fertilizer(temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous):
    input_data = [[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]]
    prediction = fertilizer_model.predict(input_data)
    return prediction[0]

# Button to get fertilizer prediction
if st.button("🧪 Recommend Fertilizer"):
    recommended_fertilizer = recommend_fertilizer(temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous)
    st.success(f"🧪 Recommended Fertilizer: **{recommended_fertilizer}**")
