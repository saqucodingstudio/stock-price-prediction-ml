# Stock Price Prediction using Machine Learning

An educational time-series forecasting project that compares a naive baseline,
Linear Regression, and Random Forest on historical daily stock-market data.

## Important data note
The provided dataset's company/ticker and source have not been verified. Do not
label it as TCS or another company until the source is confirmed. This project is
for learning and is not investment advice.

## Dataset
Place the CSV at `data/stock_market.csv`. Required columns:
`Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
Additional columns such as `Adj Close`, `Dividends`, and `Stock Splits` are allowed.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the experiment
```powershell
python main.py
```
Outputs are saved in `outputs/`:
- `model_comparison.csv`
- `predictions.csv`
- `actual_vs_predicted.png`
- `residuals.png`

## Run the dashboard
```powershell
streamlit run dashboard.py
```
The dashboard loads `data/stock_market.csv` by default, or accepts a CSV upload.

## Method
- Sort rows by date.
- Create `Target` from the next observed trading session's `Close`.
- Use Open, High, Low, Close, and Volume as features.
- Split chronologically; earlier dates train, later dates test.
- Compare a naive baseline (`next close = current close`) with Linear Regression
  and Random Forest.
- Report MAE, RMSE, and R² on the held-out period.

Because same-session OHLCV features are used, this is a forecast made after that
session closes. For a forecast available before market open, the feature design
would need to use only information available at that time. Test performance does
not guarantee future performance.
