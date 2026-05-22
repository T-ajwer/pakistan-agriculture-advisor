import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data and train model
df = pd.read_csv("Crop_recommendation.csv")
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# App UI
st.title("🌾 KISSAN Crop Recommendation System")
st.markdown("Enter your soil and weather conditions to get the best crop recommendation!!")

N = st.slider("Nitrogen (N)", 0, 140, 50)
P = st.slider("Phosphorus (P)", 5, 145, 50)
K = st.slider("Potassium (K)", 5, 205, 50)
temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
ph = st.slider("pH Level", 3.5, 10.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("🌱 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    probabilities = rf_model.predict_proba(input_data)[0]
    classes = rf_model.classes_
    
    # Get top 3
    top3_indices = probabilities.argsort()[-3:][::-1]
    
    st.markdown("### 🌾 Top Crop Recommendations")
    for i, idx in enumerate(top3_indices):
        crop = classes[idx].upper()
        confidence = probabilities[idx] * 100
        if i == 0:
            st.success(f"🥇 **{crop}** — {confidence:.1f}% confidence")
        elif i == 1:
            st.info(f"🥈 **{crop}** — {confidence:.1f}% confidence")
        else:
            st.warning(f"🥉 **{crop}** — {confidence:.1f}% confidence")