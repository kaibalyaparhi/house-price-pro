import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="House AI", layout="wide")

# ================= ULTRA PREMIUM CSS =================
st.markdown("""
<style>

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #ff6a00, #ee0979);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

/* Animation */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass card */
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Title */
h1 {
    text-align: center;
    color: white;
    font-size: 40px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.6);
}

/* Inputs */
label {
    color: white !important;
    font-weight: bold;
}

/* Button glow */
.stButton>button {
    background: linear-gradient(45deg, #ff6a00, #ee0979);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    box-shadow: 0 0 20px rgba(255,105,135,0.7);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* Success box */
.stSuccess {
    background: rgba(0,255,100,0.2);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD =================
df = pd.read_csv("Housing.csv")
model = pickle.load(open("house_model.pkl", "rb"))

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Controls")

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

# ================= MAIN UI =================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.title("🏠 House Price AI Predictor")

st.markdown("### 💡 Smart prediction powered by Machine Learning")

# Prediction
if st.button("🚀 Predict Price"):
    data = pd.DataFrame([[area, bedrooms, bathrooms, stories,
                          mainroad, guestroom, basement, parking]],
                        columns=['area', 'bedrooms', 'bathrooms', 'stories',
                                 'mainroad', 'guestroom', 'basement', 'parking'])

    result = model.predict(data)[0]

    st.success(f"💰 Estimated Price: ₹ {round(result, 2)}")

st.markdown('</div>', unsafe_allow_html=True)

# ================= DASHBOARD =================
st.markdown("## 📊 Insights Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Area vs Price")
    st.scatter_chart(df[['area', 'price']])

with col2:
    st.subheader("Price Distribution")
    st.bar_chart(df['price'])