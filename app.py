
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ================= PAGE =================
st.set_page_config(page_title="🏠 House AI Dashboard", layout="wide")

# ================= PREMIUM UI =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🏠 House Price AI Dashboard 🚀")

# ================= LOAD MODEL =================
try:
    model = pickle.load(open("house_model.pkl", "rb"))
except:
    st.error("Model load failed")
    st.stop()

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["💰 Prediction", "📊 Data Insights", "🔍 Explain AI"])

# ================= TAB 1 =================
with tab1:

    st.markdown("## Enter House Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area", 500, 10000, 2000)
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)

    with col2:
        stories = st.number_input("Stories", 1, 5, 2)
        parking = st.number_input("Parking", 0, 5, 1)

    if st.button("🚀 Predict Price"):
        data = np.array([[area, bedrooms, bathrooms, stories, parking]])
        result = model.predict(data)[0]
        st.success(f"💰 Price: ₹ {round(result,2)}")

# ================= LOAD DATA =================
try:
    df = pd.read_csv("Housing.csv")

    if 'area' not in df.columns:
        columns = [
            'price','area','bedrooms','bathrooms','stories',
            'mainroad','guestroom','basement','hotwaterheating',
            'airconditioning','parking','prefarea','furnishingstatus'
        ]
        df = pd.read_csv("Housing.csv", names=columns, header=None)

except:
    df = None

# ================= TAB 2 =================
with tab2:

    st.markdown("## 📊 Data Insights")

    if df is not None:
        st.subheader("Area vs Price")
        st.scatter_chart(df[['area','price']])

        st.subheader("Distribution")
        st.bar_chart(df[['bedrooms','bathrooms','stories']])

    else:
        st.warning("Dataset not loaded")

# ================= TAB 3 =================
with tab3:

    st.markdown("## 🔍 Explainable AI (SHAP)")

    if st.button("Show SHAP Explanation"):

        try:
            sample = np.array([[2000, 3, 2, 2, 1]])

            explainer = shap.Explainer(model)
            shap_values = explainer(sample)

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP error: {e}")

# ================= FEATURE IMPORTANCE =================
st.markdown("---")
st.markdown("## 📈 Feature Importance")

try:
    fi = pd.read_csv("feature_importance.csv")
    st.bar_chart(fi.set_index("feature"))
except:
    st.info("No feature importance file")

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 Built by Kaibalya | AI Dashboard")