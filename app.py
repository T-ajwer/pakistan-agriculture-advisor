import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data and train model
df = pd.read_csv("Crop_recommendation.csv")
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(rf_model)
feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']

# UI
st.title("🌾 Pakistan Agriculture Advisor")
st.markdown("Enter your soil and climate conditions to see what your land can grow.")

N = st.slider("Nitrogen (N)", 0, 140, 50)
P = st.slider("Phosphorus (P)", 5, 145, 50)
K = st.slider("Potassium (K)", 5, 205, 50)
temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
ph = st.slider("pH Level", 3.5, 10.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("🌱 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_df = pd.DataFrame(input_data, columns=X.columns)

    probabilities = rf_model.predict_proba(input_data)[0]
    classes = rf_model.classes_
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

    # SHAP section
    st.markdown("---")
    top_crop = classes[top3_indices[0]].upper()
    st.markdown(f"### 🔍 Why did the model recommend {top_crop}?")

    shap_values = explainer.shap_values(input_df)
top_class_idx = top3_indices[0]

# Handles both old and new SHAP output formats
if isinstance(shap_values, list):
    shap_vals = shap_values[top_class_idx][0]   # old SHAP format
else:
    shap_vals = shap_values[0, :, top_class_idx]  # new SHAP format

    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in shap_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, shap_vals, color=colors)
    ax.axvline(x=0, color='white', linewidth=0.8)
    ax.set_xlabel('SHAP Value — Impact on Recommendation', color='white')
    ax.set_title(f'What drove the {top_crop} recommendation', color='white')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    st.pyplot(fig)
    plt.close()

    st.caption("🟢 Green = pushed toward this crop   |   🔴 Red = pushed against it")