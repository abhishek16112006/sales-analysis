import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Startup Dashboard", layout="wide")

# =========================
# CUSTOM CSS (🔥 PRO UI)
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🚀 E-Commerce Analytics Dashboard")
st.markdown("### Real-time Sales Insights & Prediction")

# =========================
# LOAD DATA + MODEL
# =========================
base_path = os.path.dirname(__file__)



df = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\sales-analysis\\E-Commerce Sales Analysis\\data\\sales.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"] = df["Order Date"].dt.month

with open("C:\\Users\\DELL\\OneDrive\\Desktop\\sales-analysis\\E-Commerce Sales Analysis\\model\\sales_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# SIDEBAR FILTERS 🎛
# =========================
st.sidebar.header("🔍 Filters")

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[df["Category"].isin(category_filter)]

# =========================
# KPI CARDS 💰
# =========================
total_sales = int(filtered_df["Sales"].sum())
total_profit = int(filtered_df["Profit"].sum())
total_orders = len(filtered_df)

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"₹{total_sales}")
col2.metric("📈 Total Profit", f"₹{total_profit}")
col3.metric("📦 Total Orders", total_orders)

st.markdown("---")

# =========================
# PREDICTION 🔮
# =========================
st.subheader("🔮 Predict Future Sales")

c1, c2 = st.columns(2)

with c1:
    month = st.slider("Month", 1, 12)

with c2:
    quantity = st.number_input("Quantity", 1, 100)

if st.button("Predict"):
    pred = model.predict([[month, quantity]])
    st.success(f"Predicted Sales: ₹{pred[0]:.2f}")

st.markdown("---")

# =========================
# CHARTS 📊
# =========================
col1, col2 = st.columns(2)

# Sales Trend
with col1:
    st.markdown("### 📈 Sales Trend")
    fig, ax = plt.subplots()
    sns.lineplot(x="Month", y="Sales", data=filtered_df, marker="o", ax=ax)
    st.pyplot(fig)

# Category Sales
with col2:
    st.markdown("### 🛒 Category Sales")
    fig, ax = plt.subplots()
    sns.barplot(x="Category", y="Sales", data=filtered_df, ax=ax)
    st.pyplot(fig)

# =========================
# SECOND ROW
# =========================
col3, col4 = st.columns(2)

# Profit Distribution
with col3:
    st.markdown("### 💹 Profit Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Profit"], kde=True, ax=ax)
    st.pyplot(fig)

# Heatmap
with col4:
    st.markdown("### 🔥 Correlation")
    fig, ax = plt.subplots()
    sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, ax=ax)
    st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("✨ Built like a Startup Dashboard using Seaborn + Streamlit")