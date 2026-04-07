import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sqlite3

# ================= PAGE =================
st.set_page_config(page_title="🏠 House AI Dashboard", layout="wide")

# ================= LOGIN =================
def login():
    st.title("🔐 Login System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "Amit" and password == "1234":
            st.session_state["login"] = True
            st.success("Login Successful ✅")
            st.rerun()   # 🔥 FIX
        else:
            st.error("Invalid Credentials ❌")

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
    st.stop()

# ================= DATABASE =================
conn = sqlite3.connect("data.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    area REAL,
    bedrooms INT,
    bathrooms INT,
    stories INT,
    parking INT,
    price REAL
)
""")

# ================= UI =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 House Price AI Dashboard 🚀")

# ================= LOAD MODEL =================
try:
    model = pickle.load(open("house_model.pkl", "rb"))
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Prediction",
    "📊 Data Insights",
    "🔍 Explain AI",
    "📜 History"
])

# ================= TAB 1 =================
with tab1:

    st.subheader("Enter House Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area", 500, 10000, 2000)
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)

    with col2:
        stories = st.number_input("Stories", 1, 5, 2)
        parking = st.number_input("Parking", 0, 5, 1)

    if st.button("🚀 Predict Price"):
        try:
            data = np.array([[area, bedrooms, bathrooms, stories, parking]])
            result = model.predict(data)[0]

            st.success(f"💰 Price: ₹ {round(result,2)}")

            # Save to DB
            c.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                      (area, bedrooms, bathrooms, stories, parking, result))
            conn.commit()

        except Exception as e:
            st.error(f"Prediction error: {e}")

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

    st.subheader("📊 Data Insights")

    if df is not None:
        st.write("Area vs Price")
        st.scatter_chart(df[['area','price']])

        st.write("Feature Distribution")
        st.bar_chart(df[['bedrooms','bathrooms','stories']])
    else:
        st.warning("Dataset not loaded")

# ================= TAB 3 =================
with tab3:

    st.subheader("🔍 Explainable AI (SHAP)")

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

# ================= TAB 4 =================
with tab4:

    st.subheader("📜 Prediction History")

    try:
        data = pd.read_sql("SELECT * FROM predictions", conn)
        st.dataframe(data)
    except:
        st.warning("No history yet")

# ================= FEATURE IMPORTANCE =================
st.markdown("---")
st.subheader("📈 Feature Importance")

try:
    fi = pd.read_csv("feature_importance.csv")
    st.bar_chart(fi.set_index("feature"))
except:
    st.info("Feature importance not available")

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 Built by Kaibalya | Full Stack ML App")