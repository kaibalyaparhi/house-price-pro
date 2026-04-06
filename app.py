import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="House AI", layout="wide")

# ================= LOAD =================
try:
    model = pickle.load(open("house_model.pkl","rb"))
except:
    st.error("❌ Model not found")
    st.stop()

# ================= UI =================
st.title("🏠 House Price Prediction AI")

st.sidebar.header("Enter Features")

area = st.sidebar.slider("Area", 0, 10000, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 0, 10, 2)
stories = st.sidebar.slider("Stories", 0, 5, 2)
parking = st.sidebar.slider("Parking", 0, 5, 1)

# ================= PREDICT =================
if st.button("🚀 Predict"):

    data = np.array([[area, bedrooms, bathrooms, stories, parking]])

    result = model.predict(data)[0]

    st.success(f"💰 Price: ₹ {round(result, 2)}")

    # ================= SHAP =================
    st.markdown("## 🧠 Model Explanation")

    try:
        import shap

        explainer = shap.Explainer(model)
        shap_values = explainer(data)

        shap.plots.bar(shap_values)
        st.pyplot()

    except:
        st.info("SHAP not supported for this model")

# ================= FEATURE IMPORTANCE =================
st.markdown("## 🔍 Feature Importance")

try:
    fi = pd.read_csv("feature_importance.csv")
    st.bar_chart(fi.set_index("feature"))
except:
    st.info("Feature importance not available")