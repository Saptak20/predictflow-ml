import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"
FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_NAMES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "model.pkl not found. Add your trained model to the project root."
        )
    return joblib.load(path)


st.set_page_config(page_title="PredictFlow ML", page_icon="🔍", layout="centered")
st.title("PredictFlow ML - Iris Prediction")
st.write("Enter flower measurements and click Predict.")

with st.form("prediction_form"):
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=15.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    predict_clicked = st.form_submit_button("Predict")

if predict_clicked:
    try:
        model = load_model(MODEL_PATH)

        user_input = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=FEATURES,
        )

        numeric_input = user_input.astype(float)
        prediction = model.predict(numeric_input)[0]

        predicted_label = TARGET_NAMES.get(int(prediction), f"Class {prediction}")
        st.success(f"Prediction: {predicted_label}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(numeric_input)[0]
            prob_df = pd.DataFrame(
                {
                    "Class": [TARGET_NAMES.get(i, f"Class {i}") for i in range(len(probs))],
                    "Probability": probs,
                }
            )
            st.subheader("Prediction Confidence")
            st.dataframe(prob_df, use_container_width=True)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
