import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Title
st.title("🔋 Laptop Battery Life Predictor")
st.markdown("Estimate your battery life based on brightness, tabs, and background apps.")

# --- Simulated Training Data ---
np.random.seed(0)
n = 300
brightness = np.random.uniform(10, 100, n).reshape(-1, 1)
tabs = np.random.randint(1, 20, n).reshape(-1, 1)
apps = np.random.randint(0, 10, n).reshape(-1, 1)

# Simulated formula: non-linear with noise
battery = (
    -0.02 * brightness**2
    + 1.5 * brightness
    - 0.8 * tabs
    - 1.2 * apps
    + 120
    + np.random.randn(n, 1) * 2
)

# Prepare dataset
X = np.hstack((brightness, tabs, apps))
y = battery

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# --- User Inputs ---
brightness_input = st.slider("💡 Screen Brightness (%)", 10, 100, 50)
tabs_input = st.slider("🌐 Browser Tabs Open", 1, 20, 5)
apps_input = st.slider("⚙️ Background Apps Running", 0, 10, 2)

# Predict
user_data = np.array([[brightness_input, tabs_input, apps_input]])
user_poly = poly.transform(user_data)
prediction = model.predict(user_poly)

# --- Output ---
st.subheader("🔮 Predicted Battery Life")
st.success(f"{prediction[0][0]:.2f} hours")

# Optional: Show inputs
with st.expander("📋 Show Input Details"):
    st.write({
        "Brightness (%)": brightness_input,
        "Tabs Open": tabs_input,
        "Background Apps": apps_input
    })
