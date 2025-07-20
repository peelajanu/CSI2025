import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="California House Price Prediction", layout="wide")

model = joblib.load("house_price_model.pkl")
data = pd.read_csv("california_housing.csv")  

st.title("🏡 California House Price Prediction App")
st.markdown("Enter the features below to predict house price (in 100,000s USD).")

st.sidebar.header("🔧 Input Features")
MedInc = st.sidebar.slider("Median Income (10k USD)", 0.0, 20.0, 3.0)
HouseAge = st.sidebar.slider("House Age", 1, 52, 20)
AveRooms = st.sidebar.slider("Average Rooms per Household", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms per Household", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population in Block", 3, 36062, 1000)
AveOccup = st.sidebar.slider("Average Occupants per Household", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -118.0)

input_data = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude]
})
prediction = model.predict(input_data)[0]

st.subheader("📊 User Input:")
st.dataframe(input_data)

st.success(f"💰 **Predicted House Price:** ${prediction * 100000:,.2f}")

st.subheader("📈 Feature Distributions (with your input)")

col1, col2 = st.columns(2)

with col1:
    selected_feature = st.selectbox("Select a feature to compare:", input_data.columns)
    fig, ax = plt.subplots()
    sns.histplot(data[selected_feature], kde=True, bins=30, ax=ax, color='lightblue')
    ax.axvline(input_data[selected_feature][0], color='red', linestyle='--', label='Your Input')
    ax.legend()
    ax.set_title(f"{selected_feature} Distribution")
    st.pyplot(fig)

with col2:
    st.markdown("### 🔍 Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig2, ax2 = plt.subplots()
        sns.barplot(data=importance, x="Importance", y="Feature", palette="viridis", ax=ax2)
        ax2.set_title("Feature Importance")
        st.pyplot(fig2)
    else:
        st.warning("Feature importance not available for this model.")
