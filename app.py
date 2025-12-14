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
import io
import joblib
import requests
import os

# ===================== CONFIG (defaults) =====================

# Where predictions (JSON) are stored
DEFAULT_BUCKET = "cloudprojectmodel"
DEFAULT_PREFIX = "predictions/"

# Your API Gateway endpoint that calls lambda_store_predictions
DEFAULT_API_URL = "https://zmjbu0xzc7.execute-api.us-east-1.amazonaws.com/Prod/predict-stream"

# Where your two trained models (.pkl) are stored in S3
DEFAULT_MODEL_BUCKET = "cloudprojectmodel"
DEFAULT_RF_MODEL_KEY = "model/student_g3_model.pkl"
DEFAULT_GB_MODEL_KEY = "model/student_g3_gb_predict.pkl"

# ===================== LOAD SECRETS (if present) =====================

# .streamlit/secrets.toml should look like:
# [aws]
# AWS_ACCESS_KEY_ID = "..."
# AWS_SECRET_ACCESS_KEY = "..."
# AWS_REGION = "us-east-1"
#
# [app]
# API_URL = "https://.../Prod/predict-stream"
# S3_BUCKET = "cloudprojectmodel"
# S3_PREFIX = "predictions/"

try:
    # AWS credentials from secrets -> environment variables for boto3
    if "aws" in st.secrets:
        aws_conf = st.secrets["aws"]
        os.environ["AWS_ACCESS_KEY_ID"] = aws_conf["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_conf["AWS_SECRET_ACCESS_KEY"]
        os.environ["AWS_DEFAULT_REGION"] = aws_conf.get("AWS_REGION", "us-east-1")

    # Optional: override defaults from [app] section
    if "app" in st.secrets:
        app_conf = st.secrets["app"]
        DEFAULT_API_URL = app_conf.get("API_URL", DEFAULT_API_URL)
        DEFAULT_BUCKET = app_conf.get("S3_BUCKET", DEFAULT_BUCKET)
        DEFAULT_PREFIX = app_conf.get("S3_PREFIX", DEFAULT_PREFIX)
except Exception:
    # If secrets missing / misconfigured, just keep defaults
    pass

# ===================== HELPERS =====================


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
def load_models(model_bucket: str, rf_key: str, gb_key: str):
    """
    Load both trained models (.pkl) from S3.
    Assumes scikit-learn models saved with joblib.
    """
    s3 = boto3.client("s3")

    def _load_one(key: str):
        obj = s3.get_object(Bucket=model_bucket, Key=key)
        byts = obj["Body"].read()
        return joblib.load(io.BytesIO(byts))

    rf_model = _load_one(rf_key)
    gb_model = _load_one(gb_key)

    return rf_model, gb_model


def generate_synthetic_student() -> dict:
    """
    Generate one synthetic student with G1, G2, and a synthetic true G3.
    True G3 is used only for RMSE comparison between the 2 models.
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

    # Simple synthetic logic linking features to grades
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
    G3 = clamp(G2 + random.randint(-2, 2))  # "true" final grade for RMSE

    student["G1"] = G1
    student["G2"] = G2
    student["G3"] = G3  # synthetic true label (not shown to admins)

    return student


def send_predictions_via_api(api_url: str, records: list) -> tuple[int, str]:
    """Send predicted records to Lambda via API Gateway (ONE API CALL)."""
    payload = {"records": records}
    resp = requests.post(api_url, json=payload, timeout=60)
    return resp.status_code, resp.text


# ===================== MAIN APP =====================


def main():
    st.title("Student G3 Prediction Dashboard (Cloud Project)")

    st.markdown(
        """
This dashboard uses **two ML models** (stored in S3) to predict students'
final grades (**G3**) from their earlier performance and background.

**Pipeline (per run):**

1. Generate **synthetic students** in Streamlit  
2. Load **two trained models** (`.pkl`) from **Amazon S3**  
3. Predict G3 with **both** models and pick the **best one by RMSE**  
4. Send all predictions in **ONE API call** to **API Gateway → Lambda**  
5. Lambda stores the results in **Amazon S3**  
6. This dashboard reads predictions from S3 and highlights **at‑risk students**
        """
    )

    # ========== SIDEBAR CONFIG ==========
    st.sidebar.header("Configuration")

    bucket = st.sidebar.text_input("S3 Bucket (predictions)", DEFAULT_BUCKET)
    prefix = st.sidebar.text_input("S3 Prefix (predictions)", DEFAULT_PREFIX)
    api_url = st.sidebar.text_input("API URL (Lambda via API Gateway)", DEFAULT_API_URL)

    model_bucket = st.sidebar.text_input("Model S3 Bucket", DEFAULT_MODEL_BUCKET)
    rf_key = st.sidebar.text_input("RandomForest model key (.pkl)", DEFAULT_RF_MODEL_KEY)
    gb_key = st.sidebar.text_input("GradientBoosting model key (.pkl)", DEFAULT_GB_MODEL_KEY)

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

    # Load both models once (cached)
    try:
        rf_model, gb_model = load_models(model_bucket, rf_key, gb_key)
        st.sidebar.markdown("Models loaded: **RandomForest**, **GradientBoosting**")
    except Exception as e:
        st.sidebar.error(f"Error loading models from S3: {e}")
        return

    # ========== DEBUG BUTTON 1: Synthetic + model.predict (no Lambda) ==========
    if st.sidebar.button("🧪 Debug: generate & predict (no Lambda)"):
        with st.spinner("Debug: generating 20 synthetic students and predicting..."):
            try:
                students = [generate_synthetic_student() for _ in range(20)]
                df_students = pd.DataFrame(students)

                st.write("Synthetic students (first 5):")
                st.dataframe(df_students.head())

                if "G3" in df_students.columns:
                    y_true = df_students["G3"].astype(float).values
                    X = df_students.drop(columns=["G3"])
                else:
                    y_true = None
                    X = df_students

                st.write("Feature columns going into model:", list(X.columns))
                st.write("Shape of X:", X.shape)

                rf_pred = rf_model.predict(X)
                gb_pred = gb_model.predict(X)

                st.write("RF predictions (first 5):", rf_pred[:5])
                st.write("GB predictions (first 5):", gb_pred[:5])

                st.success("Synthetic generation + model.predict() succeeded ✅")
            except Exception as e:
                st.error("Error during synthetic generation or model prediction:")
                st.exception(e)

    # ========== DEBUG BUTTON 2: Simple Lambda call ==========
    if st.sidebar.button("🧪 Debug: send 1 simple record to Lambda"):
        test_record = {
            "id": "debug-1",
            "G1": 10,
            "G2": 12,
            "predicted_G3": 14,
            "model_used": "DebugModel",
            "prediction_time": datetime.utcnow().isoformat(),
        }
        try:
            status_code, text = send_predictions_via_api(api_url, [test_record])
            st.write("Status code:", status_code)
            st.write("Response text:", text)
            if status_code == 200:
                st.success("Simple Lambda test succeeded ✅")
            else:
                st.error("Lambda/API returned non-200 response")
        except Exception as e:
            st.error("Error calling Lambda/API Gateway:")
            st.exception(e)

    # ========== PIPELINE BUTTON ==========
    if st.sidebar.button("🚀 Run pipeline (Generate + Compare + Send)"):
        if not bucket or not prefix or not api_url:
            st.error("Please configure S3 Bucket, Prefix, and API URL.")
        else:
            with st.spinner(
                f"Generating {int(n_new)} synthetic students and predicting G3..."
            ):
                try:
                    # 1) Generate synthetic students
                    n_new_int = int(n_new)
                    students = [generate_synthetic_student() for _ in range(n_new_int)]
                    df_students = pd.DataFrame(students)

                    # 2) Separate features (X) and true labels (y_true)
                    if "G3" in df_students.columns:
                        y_true = df_students["G3"].astype(float).values
                        X = df_students.drop(columns=["G3"])
                    else:
                        y_true = None
                        X = df_students

                    # 3) Predict with both models
                    rf_pred = rf_model.predict(X)
                    gb_pred = gb_model.predict(X)

                    rf_pred = np.clip(rf_pred, 0.0, 20.0).astype(float)
                    gb_pred = np.clip(gb_pred, 0.0, 20.0).astype(float)

                    # 4) Compute RMSE and choose best model for this batch
                    rmse_rf = rmse_gb = None
                    best_model_name = None
                    best_pred = None

                    if y_true is not None:
                        rmse_rf = float(np.sqrt(((rf_pred - y_true) ** 2).mean()))
                        rmse_gb = float(np.sqrt(((gb_pred - y_true) ** 2).mean()))

                        if rmse_rf <= rmse_gb:
                            best_model_name = "RandomForest"
                            best_pred = rf_pred
                        else:
                            best_model_name = "GradientBoosting"
                            best_pred = gb_pred
                    else:
                        # Fallback if no labels
                        best_model_name = "GradientBoosting"
                        best_pred = gb_pred

                    # 5) Build final DataFrame to send
                    df_pred = df_students.copy()
                    if y_true is not None:
                        df_pred["true_G3"] = y_true  # internal only

                    df_pred["rf_predicted_G3"] = rf_pred
                    df_pred["gb_predicted_G3"] = gb_pred
                    df_pred["predicted_G3"] = best_pred
                    df_pred["model_used"] = best_model_name
                    df_pred["prediction_time"] = datetime.utcnow().isoformat()

                    records = df_pred.to_dict(orient="records")

                    # Preview of what we send
                    st.write("Preview of first 3 records sent to Lambda:")
                    st.json(records[:3])

                    # 6) ONE API CALL to Lambda
                    status_code, text = send_predictions_via_api(api_url, records)

                    st.write("Lambda status code:", status_code)
                    st.write("Lambda response body:", text)

                    if status_code != 200:
                        st.error(
                            f"Lambda/API returned status {status_code}."
                        )
                    else:
                        st.success(
                            f"Sent {len(records)} prediction records to Lambda.\n"
                            f"Best model for this batch: **{best_model_name}**"
                        )
                        load_s3_predictions.clear()

                except Exception as e:
                    st.error("Error during pipeline (synthetic → predict → send):")
                    st.exception(e)

    # Manual reload
    if st.sidebar.button("🔄 Reload S3 data"):
        load_s3_predictions.clear()

    # ========== DASHBOARD VIEW ==========
    df = load_s3_predictions(bucket, prefix)

    if df.empty:
        st.warning(f"No prediction data found in s3://{bucket}/{prefix}")
        return

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
            help="Students with predicted G3 below the threshold.",
        )

    # Show model name at top (NOT in every row)
    if "model_used" in df.columns and not df["model_used"].dropna().empty:
        last_model = df["model_used"].dropna().iloc[-1]
        st.markdown(f"**Current model used for predictions:** {last_model}")

    st.markdown(
        f"**Current G3 warning threshold:** {thresh} &nbsp;&nbsp; "
        f"(**At‑risk:** {at_risk_count} / {total_records})"
    )

    # ----- Predicted students table (sample) -----
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

    # ----- At-risk students -----
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

        # Bar chart of lowest predicted G3
        st.markdown("#### Lowest predicted G3 (top 30)")
        fig_low = px.bar(
            at_risk_display.head(30),
            x="id"
            if "id" in at_risk_display.columns
            else at_risk_display.index.astype(str),
            y="predicted_G3",
            title="Students with lowest predicted G3",
        )
        st.plotly_chart(fig_low, use_container_width=True)

        # ----- Factors influencing low G3 -----
        st.subheader("Factors influencing low predicted G3 (at‑risk group)")

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
                "associated with lower or higher predicted G3 within the at‑risk group."
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
