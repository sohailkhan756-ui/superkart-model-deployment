import io
import os
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL", "http://backend:7860"
).rstrip("/")

st.title("SuperKart Sales Prediction")
st.write("Enter product and store details to estimate revenue.")
st.caption("Store age uses the project's 2025 reference year.")

with st.form("single_prediction"):
    weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    sugar = st.selectbox(
        "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
    )
    area = st.number_input(
        "Product Allocated Area", min_value=0.0,
        max_value=1.0, value=0.027, format="%.4f"
    )
    mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
    size = st.selectbox("Store Size", ["Medium", "Small", "High"])
    city = st.selectbox(
        "Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"]
    )
    store_type = st.selectbox(
        "Store Type",
        ["Supermarket Type2", "Supermarket Type1",
         "Departmental Store", "Food Mart", "Supermarket Type3"]
    )
    prefix = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
    age = st.number_input("Store Age in Years", min_value=0, value=16)
    category = st.selectbox(
        "Product Type Category", ["Non Perishables", "Perishables"]
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Product_Weight": weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": area,
        "Product_MRP": mrp,
        "Store_Size": size,
        "Store_Location_City_Type": city,
        "Store_Type": store_type,
        "Product_Id_char": prefix,
        "Store_Age_Years": age,
        "Product_Type_Category": category
    }
    try:
        response = requests.post(
            BACKEND_URL + "/v1/predict", json=payload, timeout=60
        )
        response.raise_for_status()
        st.success(f"Predicted revenue: {response.json()['Sales']:,.2f}")
    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")

st.warning(
    "Supermarket Type3 was absent from training. Predictions for "
    "this category require review. New store combinations are also unvalidated."
)

st.subheader("Batch Prediction")
st.write("Upload a CSV containing the ten model input columns.")
uploaded_file = st.file_uploader("Choose CSV", type=["csv"])

if uploaded_file is not None and st.button("Predict for Batch"):
    try:
        raw = uploaded_file.getvalue()
        frame = pd.read_csv(io.BytesIO(raw))

        response = requests.post(
            BACKEND_URL + "/v1/predictbatch",
            files={"file": ("batch.csv", raw, "text/csv")},
            timeout=60
        )
        response.raise_for_status()
        predictions = response.json()

        frame["Predicted_Sales"] = [
            predictions[str(i)] for i in range(len(frame))
        ]
        st.session_state["batch_results"] = frame
    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Batch prediction failed: {error}")

if "batch_results" in st.session_state:
    results = st.session_state["batch_results"]
    st.success(f"Predictions completed for {len(results)} records")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download Predictions",
        results.to_csv(index=False),
        "superkart_predictions.csv",
        "text/csv"
    )
