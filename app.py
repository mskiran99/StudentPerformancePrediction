# app.py
import streamlit as st
import boto3
import json
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import random
import uuid
from datetime import datetime

# ---------- CONFIG ----------
DEFAULT_BUCKET = "cloudprojectmodel"
DEFAULT_PREFIX = "predictions/"
DEFAULT_API_URL = "https://zmjbu0xzc7.execute-api.us-east-1.amazonaws.com/Prod/predict-stream"
# -----------------------------


# ---------- AWS SESSION (for Streamlit Cloud) ----------
def get_s3_client():
    """
    If running on Streamlit Cloud with secrets configured, use those.
    Otherwise fall back to default boto3 client (for local/SageMaker).
    """
    if "aws" in st.secrets:
        aws_cfg = st.secrets["aws"]
        session = boto3.Session(
            aws_access_key_id=aws_cfg["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=aws_cfg["AWS_SECRET_ACCESS_KEY"],
            region_name=aws_cfg.get("AWS_REGION", "us-east-1"),
        )
        return session.client("s3")
    else:
        # local environment: use default credentials
        return boto3.client("s3")


@st.cache_data
def load_s3_predictions(bucket: str, prefix: str) -> pd.DataFrame:
    s3 = get_s3_client()
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


def generate_synthetic_student() -> dict:
    # simple version – can reuse the same logic as streaming_generator_http
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
    G3 = clamp(G2 + random.randint(-1, 1))

    student["G1"] = G1
    student["G2"] = G2
    student["G3"] = G3
    return student


def call_api_for_student(api_url: str, student: dict, model_name: str):
    payload = {"model_name": model_name, "student": student}
    resp = requests.post(api_url, json=payload)
    return resp.status_code, resp.text


def main():
    st.title("Cloud G3 Prediction Dashboard")
    st.markdown(
        """
End‑to‑end pipeline:

1. **Synthetic students** → API Gateway → Lambda  
2. Lambda writes predictions to **S3**  
3. This dashboard loads S3 data and **automatically picks the best model**
        """
    )

    # ------- Sidebar -------
    st.sidebar.header("Settings")
    bucket = st.sidebar.text_input("S3 Bucket", DEFAULT_BUCKET)
    prefix = st.sidebar.text_input("S3 Prefix", DEFAULT_PREFIX)
    api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL)

    thresh = st.sidebar.slider(
        "G3 warning threshold",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.5,
    )

    n_new = st.sidebar.number_input(
        "New synthetic students per model",
        min_value=1,
        max_value=200,
        value=10,
        step=1,
    )

    if st.sidebar.button("Generate via API"):
        if not api_url:
            st.error("Please enter API URL.")
        else:
            with st.spinner("Sending synthetic students to API..."):
                for _ in range(n_new):
                    student = generate_synthetic_student()
                    for model_name in ["RandomForest", "GradientBoosting"]:
                        call_api_for_student(api_url, student, model_name)
            st.success(f"Generated {n_new} students × 2 models = {2*n_new} requests.")
            load_s3_predictions.clear()

    if st.sidebar.button("Reload S3 data"):
        load_s3_predictions.clear()

    # ------- Load S3 data -------
    df = load_s3_predictions(bucket, prefix)
    if df.empty:
        st.warning(f"No prediction data in s3://{bucket}/{prefix}")
        return

    st.subheader("Raw predictions (first 20 rows)")
    st.dataframe(df.head(20))

    # ------- Automatic best model selection -------
    if "model_used" in df.columns and "predicted_G3" in df.columns:
        # figure out the true label column
        true_col = None
        for c in ["true_G3", "G3"]:
            if c in df.columns:
                true_col = c
                break

        if true_col:
            st.subheader("RMSE per model (computed from S3 data)")

            metrics = []
            for model_name, group in df.groupby("model_used"):
                g = group.dropna(subset=["predicted_G3", true_col])
                if g.empty:
                    continue
                mse = ((g["predicted_G3"] - g[true_col]) ** 2).mean()
                rmse = float(np.sqrt(mse))
                metrics.append(
                    {
                        "model": model_name,
                        "rmse": rmse,
                        "n_samples": len(g),
                    }
                )

            if metrics:
                metrics_df = pd.DataFrame(metrics).sort_values("rmse")
                best_model = metrics_df.iloc[0]["model"]
                st.success(
                    f"Best model **right now (based on S3 data)**: **{best_model}**"
                )
                st.dataframe(metrics_df.reset_index(drop=True))

                fig_rmse = px.bar(
                    metrics_df,
                    x="model",
                    y="rmse",
                    title="RMSE by model (lower is better)",
                    text="rmse",
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
            else:
                st.info("No rows with both predicted_G3 and true G3 to compute RMSE.")
        else:
            st.info(
                "No true G3 column (true_G3 / G3) found in S3 data. "
                "RMSE per model cannot be computed."
            )

        if "predicted_G3" in df.columns:
            st.subheader("Predicted G3 distribution by model")
            fig_dist = px.violin(
                df,
                x="model_used",
                y="predicted_G3",
                box=True,
                points="all",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info(
            "Need 'model_used' and 'predicted_G3' columns in S3 data "
            "to compare models."
        )

    # ------- At-risk students & factors -------
    if "predicted_G3" in df.columns:
        st.subheader(f"Students at risk (predicted G3 < {thresh})")
        at_risk = df[df["predicted_G3"] < thresh].copy()
        if at_risk.empty:
            st.success("No at‑risk students under this threshold.")
        else:
            st.write(f"Total at‑risk students: {len(at_risk)}")
            st.dataframe(at_risk.sort_values("predicted_G3").head(50))

            fig_low = px.bar(
                at_risk.sort_values("predicted_G3").head(30),
                x="id" if "id" in at_risk.columns else at_risk.index.astype(str),
                y="predicted_G3",
                color="model_used" if "model_used" in at_risk.columns else None,
                title="Lowest predicted G3 (top 30)",
            )
            st.plotly_chart(fig_low, use_container_width=True)

            st.subheader("Factors influencing low G3 (correlation)")
            numeric_cols = at_risk.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ["predicted_G3"]]

            if numeric_cols:
                corr = at_risk[numeric_cols + ["predicted_G3"]].corr()["predicted_G3"].drop(
                    "predicted_G3"
                )
                corr = corr.sort_values(key=lambda x: -np.abs(x))

                corr_df = corr.reset_index()
                corr_df.columns = ["feature", "corr_with_predicted_G3"]

                st.dataframe(corr_df.head(15))
                fig_corr = px.bar(
                    corr_df.head(15),
                    x="feature",
                    y="corr_with_predicted_G3",
                    title="Correlation with predicted G3 (at‑risk subset)",
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.error("No 'predicted_G3' column found in S3 data.")


if __name__ == "__main__":
    main()
