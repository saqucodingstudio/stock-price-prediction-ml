import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from stock_utils import FEATURES, load_and_prepare_data, run_experiment

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction")
st.caption(
    "Educational forecasting experiment using historical OHLCV data. "
    "The stock ticker/source is unverified; results are not investment advice."
)

with st.sidebar:
    st.header("Settings")
    csv_upload = st.file_uploader("Upload stock CSV (optional)", type=["csv"])
    test_fraction = st.slider("Test-set fraction", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    n_estimators = st.slider("Random Forest trees", min_value=100, max_value=600, value=300, step=100)
    st.write("Required columns: Date, Open, High, Low, Close, Volume")

try:
    if csv_upload is not None:
        # Read upload into a temporary in-memory CSV path is unnecessary; pass DataFrame via
        # a temporary file created by Streamlit's runtime.
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_upload.getvalue())
            temp_path = Path(tmp.name)
        try:
            df = load_and_prepare_data(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        df = load_and_prepare_data()

    results = run_experiment(df, test_fraction=test_fraction, n_estimators=n_estimators)
except Exception as exc:
    st.error(str(exc))
    st.info("Place your dataset at data/stock_market.csv, or upload a CSV using the sidebar.")
    st.stop()

first_col, second_col, third_col = st.columns(3)
first_col.metric("Usable observations", f"{len(df):,}")
second_col.metric("First date", str(df["Date"].min().date()))
third_col.metric("Last date", str(df["Date"].max().date()))

st.subheader("Historical closing price")
hist_fig, hist_ax = plt.subplots(figsize=(12, 4))
hist_ax.plot(df["Date"], df["Close"], linewidth=1.5)
hist_ax.set_xlabel("Date")
hist_ax.set_ylabel("Close (dataset units)")
hist_ax.grid(True, alpha=0.3)
hist_fig.tight_layout()
st.pyplot(hist_fig)
plt.close(hist_fig)

st.subheader("Model comparison on the held-out future period")
st.write("The split is chronological: earlier observations train the models; later observations test them.")
st.dataframe(results["metrics"].round(4), use_container_width=True, hide_index=True)
st.caption("Lower MAE and RMSE indicate smaller errors. R² can be negative on test data; it is not a percentage accuracy score.")

predictions = results["predictions"]
st.subheader("Actual vs predicted")
model_options = ["Naive baseline", "Linear Regression", "Random Forest"]
selected_models = st.multiselect(
    "Lines to display",
    options=model_options,
    default=model_options,
)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(predictions["Date"], predictions["Actual"], label="Actual", linewidth=2.2)
for model_name in selected_models:
    ax.plot(predictions["Date"], predictions[model_name], label=model_name, alpha=0.85)
ax.set_xlabel("Date")
ax.set_ylabel("Price (dataset units)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Test predictions")
st.dataframe(predictions.tail(30).round(3), use_container_width=True, hide_index=True)

st.download_button(
    "Download model comparison CSV",
    data=results["metrics"].to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv",
)
st.download_button(
    "Download test predictions CSV",
    data=predictions.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)

with st.expander("About this experiment"):
    st.markdown(
        """
        - Target: next observed trading session's closing price.
        - Features: same-session Open, High, Low, Close, and Volume.
        - Timing: these features are available only after the current session closes,
          so this is a forecast made after that session, not before market open.
        - The naive baseline predicts the next close as the current close.
        - No random shuffling is used for train/test splitting.
        - This prototype does not account for transaction costs, news, market impact,
          or all potential changes in market structure.
        """
    )
