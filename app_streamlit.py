import streamlit as st
import numpy as np
import pickle
from fpdf import FPDF
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))
scaler = StandardScaler()
# App title and description
st.markdown("""
    <h1 style='text-align: center; color: #E74C3C; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);'>
        🫀 Heart Disease Prediction App!
    </h1>
    <p style='text-align: center; color: #16A085; font-size: 20px; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);'>
        💡 Patient Data Entry - Predict whether a patient is at risk of heart disease using advanced machine learning.
    </p>
""", unsafe_allow_html=True)

# Sidebar with interactive sections and styling
st.sidebar.title("💖 Welcome to the Heart Disease Prediction App!")
st.sidebar.markdown(
    """
    <p style='color: #27AE60; font-size: 18px;'>
        Your health is our priority! This app uses advanced machine learning techniques to assess heart disease risk.
    </p>
    <br>
    <p style='color: #27AE60; font-size: 18px;'>
        Just enter the patient details to get a quick & reliable prediction, helping you make informed health decisions 💪
    </p>
    """,
    unsafe_allow_html=True,
)


# Step 1: Basic Information
with st.expander("Step 1: Basic Information"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])

# Step 2: Chest Pain and Blood Pressure
with st.expander("Step 2: Chest Pain and Blood Pressure"):
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130)

# Step 3: Cholesterol and Blood Sugar
with st.expander("Step 3: Cholesterol and Blood Sugar"):
    chol = st.number_input("Serum Cholestoral in mg/dl (chol)", min_value=50, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])

# Step 4: ECG and Exercise Data
with st.expander("Step 4: ECG and Exercise Data"):
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

# Step 5: ST Depression and Slope
with st.expander("Step 5: ST Depression and Slope"):
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])

# Step 6: Fluoroscopy and Thalassemia
with st.expander("Step 6: Fluoroscopy and Thalassemia"):
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Map the sex input to 0 or 1
sex = 1 if sex == "Male" else 0

# Predict button with animated progress bar
if st.button("Predict Heart Disease Risk 🩺"):
    patient_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    scaled_data = patient_data / np.array([120, 1, 3, 200, 600, 1, 2, 250, 1, 10, 2, 4, 3])
    prediction_proba = model.predict(scaled_data)[0][0]

    # Display animated gauge chart for prediction probability
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_proba * 100,
        title={"text": "Heart Disease Risk (%)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FF0000" if prediction_proba > 0.5 else "#00FF00"}}))
    st.plotly_chart(fig)

    # Display result with motivational message
    if prediction_proba > 0.5:
        st.error(f"🚨 High Risk! Probability: {prediction_proba * 100:.2f}%")
        st.warning("Adopt a healthier lifestyle: Regular exercise, balanced diet, and routine check-ups.")
    else:
        st.success(f"✅ Low Risk! Probability: {prediction_proba * 100:.2f}%")
        st.info("Keep maintaining a healthy lifestyle!")

    # Recent predictions and tracking
    st.markdown("### 📝 Recent Predictions")
    st.write(f"Age: {age}, Sex: {'Male' if sex == 1 else 'Female'}, Risk Probability: {prediction_proba * 100:.2f}%")
