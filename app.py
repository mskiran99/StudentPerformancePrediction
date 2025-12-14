# app.py

import streamlit as st
import boto3
import json
import pandas as pd
import numpy as np
import plotly.express as px
import random
import uuid
from datetime import datetime
import joblib
import requests

# ---------- CONFIG ----------
DEFAULT_BUCKET = "cloudprojectmodel"
DEFAULT_PREFIX = "predictions/"
# Your API Gateway URL that invokes the Lambda above:
DEFAULT_API_URL = "https://zmjbu0xzc7.execute-api.us-east-1.amazonaws.com/Prod/predict-stream"

BEST_MODEL_NAME = "GradientBoosting"  # or "RandomForest"
BEST_MODEL_FILE = (
    "student_g3_gb_predict.pkl" if BEST_MODEL_NAME == "GradientBoosting"
    else "student_g3_model.pkl"
)
# -----------------------------


@st.cache_data
def load_s3_predictions(bucket: str, prefix: str) -> pd.DataFrame:
    """Load all prediction JSONs from S3 into a DataFrame."""
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    contents = resp.get("Contents", [])
    if not contents:
        return pd.DataFrame()

    rows = []
    for obj in contents:
        key = obj["Key"]
        if key.endswith("/"):
            continue
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        try:
            rows.append(json.loads(body))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "prediction_time" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    return df


@st.cache_resource
def load_best_model():
    """Load the pre-trained 'best' model from local .pkl file."""
    model = joblib.load(BEST_MODEL_FILE)
    return model, BEST_MODEL_NAME


def generate_synthetic_student() -> dict:
    """Generate one synthetic student (no true G3, only features)."""
    student = {
        "id": str(uuid.uuid4()),
        "school": random.choice(["GP", "MS"]),
        "sex": random.choice(["F", "M"]),
        "age": random.randint(15, 22),
        "address": random.choice(["U", "R"]),
        "famsize": random.choice(["LE3", "GT3"]),
        "Pstatus": random.choice(["T", "A"]),
        "Medu": random.randint(0, 4),
        "Fedu": random.randint(0, 4),
        "Mjob": random.choice(["teacher", "health", "services", "at_home", "other"]),
        "Fjob": random.choice(["teacher", "health", "services", "at_home", "other"]),
        "reason": random.choice(["home", "reputation", "course", "other"]),
        "guardian": random.choice(["mother", "father", "other"]),
        "traveltime": random.randint(1, 4),
        "studytime": random.randint(1, 4),
        "failures": random.randint(0, 4),
        "schoolsup": random.choice(["yes", "no"]),
        "famsup": random.choice(["yes", "no"]),
        "paid": random.choice(["yes", "no"]),
        "activities": random.choice(["yes", "no"]),
        "nursery": random.choice(["yes", "no"]),
        "higher": random.choice(["yes", "no"]),
        "internet": random.choice(["yes", "no"]),
        "romantic": random.choice(["yes", "no"]),
        "famrel": random.randint(1, 5),
        "freetime": random.randint(1, 5),
        "goout": random.randint(1, 5),
        "Dalc": random.randint(1, 5),
        "Walc": random.randint(1, 5),
        "health": random.randint(1, 5),
        "absences": random.randint(0, 30),
    }

    # Synthetic G1, G2
    base = random.randint(8, 18)
    penalty_failures = 1.5 * student["failures"]
    penalty_absences = 0.1 * student["absences"]
    penalty_alcohol = 0.3 * ((student["Dalc"] - 1) + (student["Walc"] - 1))
    bonus_study = 0.8 * (student["studytime"] - 2)
    score = base - penalty_failures - penalty_absences - penalty_alcohol + bonus_study

    def clamp(x):
        return max(0, min(20, int(round(x))))

    G1 = clamp(score + random.randint(-2, 2))
    G2 = clamp(G1 + random.randint(-2, 2))

    student["G1"] = G1
    student["G2"] = G2
    return student


def send_predictions_via_api(api_url: str, records: list) -> tuple[int, str]:
    """Send predicted records to Lambda via API Gateway."""
    payload = {"records": records}
    resp = requests.post(api_url, json=payload, timeout=30)
    return resp.status_code, resp.text


def main():
    st.title("Student G3 Prediction Dashboard (Cloud Project)")

    st.markdown(
        """
**End-to-end Cloud Architecture**

- Streamlit (this app) generates synthetic students
- A pre-trained **ML model (.pkl)** predicts final grade G3
- Predictions are sent to **API Gateway → Lambda**
- Lambda stores results in **Amazon S3**
- This dashboard reads from S3 and highlights **at-risk students**
        """
    )

    # Load best model (.pkl) once
    model, best_model_name = load_best_model()

    # ------- Sidebar -------
    st.sidebar.header("Settings")

    bucket = st.sidebar.text_input("S3 Bucket", DEFAULT_BUCKET)
    prefix = st.sidebar.text_input("S3 Prefix", DEFAULT_PREFIX)
    api_url = st.sidebar.text_input("API URL (Lambda via API Gateway)", DEFAULT_API_URL)

    st.sidebar.markdown(f"Using best model: **{best_model_name}**")

    thresh = st.sidebar.slider(
        "G3 warning threshold",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.5,
        help="Students with predicted G3 below this are flagged as 'at risk'.",
    )

    n_new = st.sidebar.number_input(
        "New synthetic students per run",
        min_value=10,
        max_value=5000,
        value=1000,
        step=50,
        help="How many new students to generate and predict in one run.",
    )

    if st.sidebar.button("Generate, Predict & Send via Lambda"):
        if not bucket or not prefix or not api_url:
            st.error("Please configure S3 Bucket, Prefix, and API URL.")
        else:
            with st.spinner(
                f"Generating {int(n_new)} synthetic students and predicting G3..."
            ):
                # 1) Generate students
                n_new_int = int(n_new)
                students = [generate_synthetic_student() for _ in range(n_new_int)]
                df_students = pd.DataFrame(students)

                # 2) Predict with best model
                preds = model.predict(df_students)
                preds = np.clip(preds, 0.0, 20.0).astype(float)

                df_students["predicted_G3"] = preds
                df_students["model_used"] = best_model_name
                df_students["prediction_time"] = datetime.utcnow().isoformat()

                # 3) Send to Lambda via API Gateway
                records = df_students.to_dict(orient="records")
                status_code, text = send_predictions_via_api(api_url, records)

                if status_code != 200:
                    st.error(
                        f"Lambda/API returned status {status_code}. "
                        f"Response: {text}"
                    )
                else:
                    st.success(
                        f"Sent {len(records)} prediction records to Lambda. "
                        f"Lambda will write them to s3://{bucket}/{prefix}"
                    )
                    # Clear cache so next load reads latest S3 data
                    load_s3_predictions.clear()

    if st.sidebar.button("Reload S3 data"):
        load_s3_predictions.clear()

    # ------- Load S3 data -------
    df = load_s3_predictions(bucket, prefix)

    if df.empty:
        st.warning(f"No prediction data found in s3://{bucket}/{prefix}")
        return

    # ------- Top-level metrics -------
    st.subheader("Overview")

    if "predicted_G3" not in df.columns:
        st.error("No 'predicted_G3' column found in S3 data.")
        return

    total_records = len(df)
    at_risk_df = df[df["predicted_G3"] < thresh]
    at_risk_count = len(at_risk_df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total predicted students", total_records)
    with col2:
        st.metric(
            "At‑risk students",
            f"{at_risk_count} / {total_records}",
            help="Number of students with predicted G3 below the threshold.",
        )

    st.markdown(
        f"**Current G3 warning threshold:** {thresh} &nbsp;&nbsp; "
        f"(**At‑risk:** {at_risk_count} / {total_records})"
    )

    # ------- Predicted students sample -------
    st.subheader("Predicted Students (sample)")

    base_cols = [
        c
        for c in [
            "id",
            "school",
            "sex",
            "age",
            "G1",
            "G2",
            "predicted_G3",
            "prediction_time",
        ]
        if c in df.columns
    ]
    extra_cols = [
        c for c in ["studytime", "failures", "absences", "Dalc", "Walc"]
        if c in df.columns
    ]
    display_cols = base_cols + extra_cols

    st.dataframe(
        df[display_cols].sort_values("prediction_time", ascending=False).head(50),
        use_container_width=True,
    )

    # ------- At-risk students -------
    st.subheader(f"Students at risk (predicted G3 < {thresh})")

    if at_risk_count == 0:
        st.success("No at‑risk students under the current threshold. 🎉")
    else:
        st.write(f"Total at‑risk students: **{at_risk_count}**")

        at_risk_display = at_risk_df.sort_values("predicted_G3").head(100)
        st.dataframe(
            at_risk_display[display_cols],
            use_container_width=True,
        )

        st.markdown("#### Lowest predicted G3")
        fig_low = px.bar(
            at_risk_display.head(30),
            x="id"
            if "id" in at_risk_display.columns
            else at_risk_display.index.astype(str),
            y="predicted_G3",
            title="Students with lowest predicted G3 (top 30)",
        )
        st.plotly_chart(fig_low, use_container_width=True)

        # ------- Factors influencing low G3 -------
        st.subheader("Factors influencing low predicted G3")

        numeric_cols = at_risk_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["predicted_G3"]]

        if numeric_cols:
            corr = (
                at_risk_df[numeric_cols + ["predicted_G3"]]
                .corr()["predicted_G3"]
                .drop("predicted_G3")
            )
            corr = corr.sort_values(key=lambda x: -np.abs(x))
            corr_df = corr.reset_index()
            corr_df.columns = ["feature", "corr_with_predicted_G3"]

            st.write(
                "Higher absolute correlation means that feature is more strongly "
                "associated with lower/higher predicted G3 within the at‑risk group."
            )
            st.dataframe(corr_df.head(15), use_container_width=True)

            fig_corr = px.bar(
                corr_df.head(15),
                x="feature",
                y="corr_with_predicted_G3",
                title="Top factors correlated with predicted G3 (at‑risk students)",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info(
                "Not enough numeric features to compute correlations for at‑risk students."
            )


if __name__ == "__main__":
    main()
