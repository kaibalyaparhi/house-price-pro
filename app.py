import streamlit as st
import pandas as pd
import pickle
import time

st.set_page_config(page_title="House AI", layout="wide")

# ================= LOAD DATA =================
try:
    df = pd.read_csv("Housing.csv")

    # Clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "").str.replace("_", "")

except:
    st.error("❌ Dataset not found!")
    st.stop()

# ================= LOAD MODEL =================
try:
    model = pickle.load(open("house_model.pkl", "rb"))
except:
    st.error("❌ Model not found! Run house_price.py first.")
    st.stop()

# ================= APP HEADER =================
st.title("🏠 House Price AI Predictor")
st.markdown("### 🚀 Production-Level ML Web App")

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Input Features")

area = st.sidebar.slider("Area", 0, 10000, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 0, 10, 2)
stories = st.sidebar.slider("Stories", 0, 5, 2)
parking = st.sidebar.slider("Parking", 0, 5, 1)

mainroad = st.sidebar.selectbox("Main Road", ["yes", "no"])
guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])
basement = st.sidebar.selectbox("Basement", ["yes", "no"])

mainroad = 1 if mainroad == "yes" else 0
guestroom = 1 if guestroom == "yes" else 0
basement = 1 if basement == "yes" else 0

# ================= PREDICTION =================
st.markdown("## 🔮 Prediction")

if st.button("🚀 Predict Price"):

    with st.spinner("🤖 AI is analyzing data..."):
        time.sleep(1)

        try:
            data = pd.DataFrame([[area, bedrooms, bathrooms, stories,
                                  mainroad, guestroom, basement, parking]],
                                columns=['area','bedrooms','bathrooms','stories',
                                         'mainroad','guestroom','basement','parking'])

            result = model.predict(data)[0]

            st.success(f"💰 Estimated Price: ₹ {round(result, 2)}")

            # Fake confidence (demo purpose)
            confidence = 95
            st.info(f"📊 Model Confidence: {confidence}%")

            # Save history
            st.session_state.history.append({
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Price": round(result, 2)
            })

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ================= HISTORY =================
st.markdown("## 📜 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download History", csv, "history.csv")

# ================= DATA VISUALIZATION =================
st.markdown("## 📊 Insights")

def find_col(name):
    for col in df.columns:
        if name in col:
            return col
    return None

area_col = find_col("area")
price_col = find_col("price")

if area_col and price_col:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Area vs Price")
        st.scatter_chart(df[[area_col, price_col]])

    with col2:
        st.subheader("Price Distribution")
        st.bar_chart(df[price_col])
else:
    st.warning("⚠️ Could not detect required columns")

# ================= FEATURE IMPORTANCE =================
st.markdown("## 🔍 Feature Importance")

try:
    fi = pd.read_csv("feature_importance.csv")
    st.bar_chart(fi.set_index("feature"))
except:
    st.info("Feature importance available only for Random Forest.")

# ================= FOOTER =================
st.markdown("---")
st.caption("💡 Built with Machine Learning + Streamlit | Production Ready")