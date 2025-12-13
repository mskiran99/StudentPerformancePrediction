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
FALLBACK_MODEL = "GradientBoosting"

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
        # local environment: use default credentials / IAM role
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

    # Convert time column if present
    if "prediction_time" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")

    return df


def generate_synthetic_student() -> dict:
    """
    Generate one synthetic student with realistic fields.
    (Treat G3 as the "true" hidden value; Lambda will output predicted_G3.)
    """
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

    # These go to Lambda as inputs
    student["G1"] = G1
    student["G2"] = G2
    student["G3"] = G3   # used internally; dashboard will only display predicted_G3
    return student


def call_api_for_student(api_url: str, student: dict, model_name: str):
    """
    Call API Gateway for a single student using the chosen model.
    """
    payload = {"model_name": model_name, "student": student}
    resp = requests.post(api_url, json=payload)
    return resp.status_code, resp.text


def infer_best_model_from_s3(df: pd.DataFrame) -> str | None:
    """
    Infer the best model automatically from S3 data using RMSE
    on (true_G3 or G3) vs predicted_G3.

    Returns the name of the best model, or None if cannot infer.
    """
    if "model_used" not in df.columns or "predicted_G3" not in df.columns:
        return None

    # Choose which column has the true grade
    true_col = None
    for c in ["true_G3", "G3"]:
        if c in df.columns:
            true_col = c
            break

    if true_col is None:
        # No label to compute RMSE with
        return None

    metrics = []
    for model_name, group in df.groupby("model_used"):
        g = group.dropna(subset=["predicted_G3", true_col])
        if g.empty:
            continue
        mse = ((g["predicted_G3"] - g[true_col]) ** 2).mean()
        rmse = float(np.sqrt(mse))
        metrics.append({"model": model_name, "rmse": rmse})

    if not metrics:
        return None

    metrics_df = pd.DataFrame(metrics).sort_values("rmse")
    return metrics_df.iloc[0]["model"]


def main():
    st.set_page_config(page_title="Student Performance Prediction", layout="wide")

    st.title("Student Performance Prediction Dashboard")
    st.caption("For School Administration – powered by AWS (Lambda, API Gateway, S3)")

    # ------- Sidebar -------
    st.sidebar.header("Configuration")

    bucket = st.sidebar.text_input("S3 Bucket", DEFAULT_BUCKET)
    prefix = st.sidebar.text_input("S3 Prefix", DEFAULT_PREFIX)
    api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL)

    thresh = st.sidebar.slider(
        "At-risk threshold (Predicted G3 below)",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.5,
    )

    n_new = st.sidebar.number_input(
        "New synthetic students per run",
        min_value=200,
        max_value=3000,
        value=2000,
        step=100,
        help="How many new students to simulate and predict in one click.",
    )

    # Load data *before* we decide best model
    df_raw = load_s3_predictions(bucket, prefix)

    # Infer best model name from existing S3 data
    inferred_best_model = infer_best_model_from_s3(df_raw) if not df_raw.empty else None

    # If we can't infer yet (e.g., empty bucket or no labels), fall back
    model_for_generation = inferred_best_model or FALLBACK_MODEL

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Best model (automatically selected):**")
    st.sidebar.markdown(f"`{model_for_generation}`")

    if st.sidebar.button("Generate predictions (best model only)"):
        if not api_url:
            st.error("Please enter API URL.")
        else:
            success = 0
            failed = 0
            with st.spinner(
                f"Generating {n_new} students and sending to model: {model_for_generation}..."
            ):
                for _ in range(int(n_new)):
                    student = generate_synthetic_student()
                    status_code, _ = call_api_for_student(api_url, student, model_for_generation)
                    if 200 <= status_code < 300:
                        success += 1
                    else:
                        failed += 1

            st.success(f"API calls completed. Success: {success}, Failed: {failed}")
            if failed > 0:
                st.warning(
                    "Some requests failed – only successful calls will appear in S3. "
                    "Check Lambda/API logs if the failure count is high."
                )

            load_s3_predictions.clear()
            df_raw = load_s3_predictions(bucket, prefix)
            inferred_best_model = infer_best_model_from_s3(df_raw) if not df_raw.empty else inferred_best_model
            model_for_generation = inferred_best_model or model_for_generation

    if st.sidebar.button("Reload data from S3"):
        load_s3_predictions.clear()
        df_raw = load_s3_predictions(bucket, prefix)
        inferred_best_model = infer_best_model_from_s3(df_raw) if not df_raw.empty else inferred_best_model
        model_for_generation = inferred_best_model or model_for_generation

    # ------- If still no data, stop here -------
    if df_raw.empty:
        st.warning(f"No prediction data found in s3://{bucket}/{prefix}")
        return

    # From here on: use ALL records from S3 (all models), not filtered
    df = df_raw.copy()

    # Treat as prediction-only: drop true G3 columns from what we show
    for col in ["true_G3", "G3"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ------- Build cleaned display DataFrame -------
    display_cols = []
    for c in [
        "id",
        "school",
        "sex",
        "age",
        "studytime",
        "failures",
        "absences",
        "predicted_G3",
        "prediction_time",
        # note: we intentionally skip 'model_used' to not show it row-by-row
    ]:
        if c in df.columns:
            display_cols.append(c)

    if not display_cols:
        display_cols = df.columns.tolist()

    df_display = df[display_cols].copy()

    # Rename predicted_G3 column label for clarity
    if "predicted_G3" in df_display.columns:
        df_display = df_display.rename(columns={"predicted_G3": "Predicted G3"})

    # Sort by most recent
    if "prediction_time" in df_display.columns:
        df_display = df_display.sort_values("prediction_time", ascending=False)

    # ------- Top KPIs (total records, at-risk count) -------
    total_records = len(df_display)
    if "Predicted G3" in df_display.columns:
        at_risk_mask = df_display["Predicted G3"] < thresh
        at_risk_count = int(at_risk_mask.sum())
    else:
        at_risk_mask = pd.Series([False] * len(df_display), index=df_display.index)
        at_risk_count = 0

    best_model_for_view = inferred_best_model or model_for_generation

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Model in Use", str(best_model_for_view))

    with col2:
        st.metric("Total Students (predicted)", total_records)

    with col3:
        st.metric(
            "Students At Risk",
            f"{at_risk_count} / {total_records}" if total_records > 0 else "0 / 0",
        )

    st.markdown("---")

    # ------- Overall predictions table -------
    st.subheader("Latest Predictions")
    st.dataframe(df_display.head(500), use_container_width=True)

    # ------- Grade distribution -------
    if "Predicted G3" in df_display.columns:
        st.subheader("Grade Distribution (Predicted G3)")
        fig_hist = px.histogram(
            df_display,
            x="Predicted G3",
            nbins=20,
            title="Distribution of Predicted Final Grades",
        )
        fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ------- At-Risk Students Section -------
    st.subheader(f"Students Below Threshold (Predicted G3 < {thresh})")

    if "Predicted G3" in df_display.columns:
        at_risk_df = df_display[df_display["Predicted G3"] < thresh].copy()
    else:
        at_risk_df = pd.DataFrame()

    if at_risk_df.empty:
        st.success("No students are currently below the selected G3 threshold.")
    else:
        st.info(f"Showing {len(at_risk_df)} at-risk students out of {total_records} total.")

        # Show a clean, admin-friendly table
        show_cols = [
            "id",
            "school",
            "age",
            "studytime",
            "failures",
            "absences",
            "Predicted G3",
            "prediction_time",
        ]
        show_cols = [c for c in show_cols if c in at_risk_df.columns]

        st.dataframe(
            at_risk_df[show_cols].sort_values("Predicted G3").head(300),
            use_container_width=True,
        )

        # Simple bar chart showing lowest predicted G3 values
        fig_low = px.bar(
            at_risk_df.sort_values("Predicted G3").head(30),
            x="id" if "id" in at_risk_df.columns else at_risk_df.index.astype(str),
            y="Predicted G3",
            title="Lowest Predicted G3 (Top 30 Students)",
        )
        fig_low.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_low, use_container_width=True)

    # ------- Factors Affecting Low Predicted G3 -------
    st.subheader("📌 Factors Affecting Low Predicted G3 Scores")

    if at_risk_df.empty or "Predicted G3" not in at_risk_df.columns:
        st.info("No at-risk students available — cannot compute influencing factors.")
    else:
        # Use only numeric columns
        numeric_cols = at_risk_df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target column from predictors
        if "Predicted G3" in numeric_cols:
            numeric_cols.remove("Predicted G3")

        if not numeric_cols:
            st.info("No numeric features available to compute correlations.")
        else:
            # Compute correlation with Predicted G3
            corr = at_risk_df[numeric_cols + ["Predicted G3"]].corr()["Predicted G3"].dropna()

            # Sort strongest relationship first
            corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

            corr_df = corr.reset_index()
            corr_df.columns = ["Feature", "Correlation With Predicted G3"]

            st.write("Top factors correlated with poor performance:")
            st.dataframe(corr_df.head(10), use_container_width=True)

            fig_corr = px.bar(
                corr_df.head(10),
                x="Correlation With Predicted G3",
                y="Feature",
                orientation="h",
                title="Most Influential Factors in Low Performance",
            )
            fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
