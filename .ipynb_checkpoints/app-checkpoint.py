import streamlit as st
import boto3
import json
import pandas as pd
import plotly.express as px

# ---------- CONFIG ----------
DEFAULT_BUCKET = "cloudprojectmodel"
DEFAULT_PREFIX = "predictions/"  # folder where Lambda writes JSON
# -----------------------------

@st.cache_data
def load_data(bucket: str, prefix: str) -> pd.DataFrame:
    """Load all JSON prediction files from S3 into a DataFrame."""
    s3 = boto3.client("s3")

    # list objects
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])

    if not contents:
        return pd.DataFrame()

    rows = []
    for obj in contents:
        key = obj["Key"]
        if key.endswith("/"):  # skip folder markers
            continue
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        try:
            rows.append(json.loads(body))
        except Exception:
            # skip any non-JSON / corrupted file
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # convert timestamp if present
    if "prediction_time" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")

    return df


def main():
    st.title("Student Performance Prediction Dashboard")
    st.markdown(
        "Visualizing **predicted final grade (G3)** from our cloud streaming pipeline."
    )

    # Sidebar settings
    st.sidebar.header("S3 Settings")
    bucket = st.sidebar.text_input("S3 Bucket", DEFAULT_BUCKET)
    prefix = st.sidebar.text_input("Prefix (folder)", DEFAULT_PREFIX)

    if st.sidebar.button("Reload data"):
        st.cache_data.clear()

    df = load_data(bucket, prefix)

    if df.empty:
        st.warning("No prediction data found in S3. Make sure Lambda is writing to "
                   f"`s3://{bucket}/{prefix}` and try again.")
        return

    st.subheader("Raw prediction data (first 10 rows)")
    st.dataframe(df.head(10))

    # -------- BASIC SUMMARY --------
    if "predicted_G3" in df.columns:
        st.subheader("Summary of Predicted G3")
        st.write(df["predicted_G3"].describe())
    else:
        st.error("Column 'predicted_G3' not found in data.")
        return

    # -------- CHART 1: G3 over time --------
    if "prediction_time" in df.columns and df["prediction_time"].notna().any():
        st.subheader("Predicted G3 Over Time")
        df_time = df.sort_values("prediction_time")
        fig_time = px.line(
            df_time,
            x="prediction_time",
            y="predicted_G3",
            title="Predicted Final Grade (G3) Over Time",
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # -------- CHART 2: Distribution of G3 --------
    st.subheader("Distribution of Predicted G3")
    fig_hist = px.histogram(
        df,
        x="predicted_G3",
        nbins=20,
        title="Histogram of Predicted Final Grades (G3)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # -------- CHART 3: G2 vs Predicted G3 --------
    if "G2" in df.columns:
        st.subheader("Relationship Between G2 and Predicted G3")
        fig_scatter = px.scatter(
            df,
            x="G2",
            y="predicted_G3",
            trendline="ols",
            title="G2 vs Predicted G3",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -------- CHART 4: Predicted G3 by School --------
    if "school" in df.columns:
        st.subheader("Predicted G3 by School")
        fig_box = px.box(
            df,
            x="school",
            y="predicted_G3",
            title="Predicted G3 Distribution by School (GP vs MS)",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # -------- CHART 5: Studytime & Failures Heatmap --------
    if "studytime" in df.columns and "failures" in df.columns:
        st.subheader("Average Predicted G3 by Studytime and Failures")
        fig_heat = px.density_heatmap(
            df,
            x="studytime",
            y="failures",
            z="predicted_G3",
            histfunc="avg",
            title="Studytime vs Failures vs Avg Predicted G3",
        )
        st.plotly_chart(fig_heat, use_container_width=True)


if __name__ == "__main__":
    main()
