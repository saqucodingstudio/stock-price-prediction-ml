from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED_COLUMNS = ["Date", *FEATURES]


def load_and_prepare_data(csv_path=None):
    """Load a daily OHLCV CSV and create next-trading-session Close target."""
    path = Path(csv_path) if csv_path else DATA_DIR / "stock_market.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlace your CSV at data/stock_market.csv "
            "or choose the correct CSV path in the dashboard."
        )

    df = pd.read_csv(path)
    # Normalize common whitespace in column names.
    df.columns = [str(c).strip() for c in df.columns]

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Avoid silently treating repeated trading dates as separate observations.
    duplicate_dates = int(df["Date"].duplicated().sum())
    if duplicate_dates:
        raise ValueError(
            f"Found {duplicate_dates} duplicate dates. Inspect the CSV and resolve "
            "duplicate dates before modelling."
        )

    # Next row is the next observed trading session (not the next calendar day).
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna(subset=["Target"]).reset_index(drop=True)

    if len(df) < 50:
        raise ValueError("Dataset has fewer than 50 usable rows; more data is recommended.")

    return df


def chronological_split(df, test_fraction=0.2):
    """Split in date order; never shuffle time-series observations."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5.")
    split_index = int(len(df) * (1 - test_fraction))
    if split_index < 1 or split_index >= len(df):
        raise ValueError("Invalid split; adjust the test fraction or dataset size.")

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test


def _metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def run_experiment(df, test_fraction=0.2, random_state=42, n_estimators=300):
    """Train baselines/models on earlier dates and evaluate on later dates."""
    train, test = chronological_split(df, test_fraction)

    X_train = train[FEATURES]
    y_train = train["Target"]
    X_test = test[FEATURES]
    y_test = test["Target"]

    # Naive baseline: next close is predicted as the current day's close.
    naive_pred = test["Close"].to_numpy()

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    linear_pred = linear.predict(X_test)

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)

    predictions = pd.DataFrame({
        "Date": test["Date"].to_numpy(),
        "Actual": y_test.to_numpy(),
        "Naive baseline": naive_pred,
        "Linear Regression": linear_pred,
        "Random Forest": forest_pred,
    })

    metrics_rows = []
    for name, values in [
        ("Naive baseline", naive_pred),
        ("Linear Regression", linear_pred),
        ("Random Forest", forest_pred),
    ]:
        row = {"Model": name}
        row.update(_metrics(y_test, values))
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows).sort_values("MAE").reset_index(drop=True)
    return {
        "train": train,
        "test": test,
        "metrics": metrics,
        "predictions": predictions,
        "models": {"Linear Regression": linear, "Random Forest": forest},
        "features": FEATURES,
        "test_fraction": test_fraction,
    }


def save_experiment_outputs(results):
    """Save reproducible CSV outputs; figures are made by main.py/dashboard.py."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results["metrics"].to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    results["predictions"].to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    return OUTPUT_DIR
