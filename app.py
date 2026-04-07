import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ================= PAGE =================
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")

# ================= UI =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1 {color:white;text-align:center;}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    border-radius:10px;
    height:3em;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 House Price Prediction + Explainable AI")

# ================= LOAD MODEL =================
try:
    model = pickle.load(open("house_model.pkl", "rb"))
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("📐 Area (sq ft)", 500, 10000, 2000)
    bedrooms = st.number_input("🛏 Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("🛁 Bathrooms", 1, 10, 2)

with col2:
    stories = st.number_input("🏢 Stories", 1, 5, 2)
    parking = st.number_input("🚗 Parking", 0, 5, 1)

# ================= PREDICT =================
if st.button("🚀 Predict Price"):

    try:
        data = np.array([[area, bedrooms, bathrooms, stories, parking]])
        result = model.predict(data)[0]

        st.success(f"💰 Estimated Price: ₹ {round(result, 2)}")

        # ================= SHAP =================
        st.markdown("## 🔍 SHAP Explanation")

        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(data)

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"SHAP not supported for this model: {e}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ================= DATA VIS =================
st.markdown("---")
st.markdown("## 📊 Data Visualization")

try:
    df = pd.read_csv("Housing.csv")

    # Fix broken CSV (no header case)
    if 'area' not in df.columns:
        columns = [
            'price','area','bedrooms','bathrooms','stories',
            'mainroad','guestroom','basement','hotwaterheating',
            'airconditioning','parking','prefarea','furnishingstatus'
        ]
        df = pd.read_csv("Housing.csv", names=columns, header=None)

    if 'area' in df.columns and 'price' in df.columns:
        st.scatter_chart(df[['area','price']])
    else:
        st.warning("Columns missing for visualization")

except Exception as e:
    st.warning(f"Dataset error: {e}")

# ================= FEATURE IMPORTANCE =================
st.markdown("---")
st.markdown("## 📈 Feature Importance")

try:
    fi = pd.read_csv("feature_importance.csv")
    st.bar_chart(fi.set_index("feature"))
except:
    st.info("Feature importance not available")

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 Built by Kaibalya | ML + Explainable AI")