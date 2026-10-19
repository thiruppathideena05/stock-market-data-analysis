# Stock Market Data Analysis

An intermediate-level Data Science portfolio project using public historical stock-market data, Python, Jupyter Notebook, and Streamlit.

## Features

- Download historical daily OHLCV data through `yfinance`
- Analyze closing prices, returns, trading volume, moving averages, annualized volatility, and drawdown
- Train a Random Forest regression baseline to estimate the next observed trading session's closing price
- Evaluate the model on a chronological holdout set against the naive "tomorrow equals today's close" baseline
- Explore data in a Streamlit dashboard and download CSV results

> **Disclaimer:** This project is for education and data analysis only, not investment advice. It does not guarantee accurate predictions or profitable trading. Historical results do not guarantee future performance.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, yfinance, Scikit-learn, Plotly, Streamlit, Jupyter Notebook.

## Project structure

```text
stock_market_data_analysis/
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── stock_market_analysis.ipynb
├── data/
│   └── (downloaded CSV files are saved here)
└── outputs/
    └── (analysis exports and test predictions are saved here)
```

## Requirements

- Python 3.10 or newer recommended
- Internet connection to download Yahoo Finance data
- Windows, macOS, or Linux

## Installation (Windows)

Open Command Prompt or the VS Code terminal:

```bash
cd stock_market_data_analysis
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not recognized, install Python from https://www.python.org/downloads/ and enable **Add Python to PATH** during installation.

## Run the Jupyter Notebook

From the project root:

```bash
jupyter notebook
```

Open `notebooks/stock_market_analysis.ipynb` and run cells from top to bottom.

The notebook defaults to `AAPL` and downloads five years of data. You can change `TICKER` to `MSFT`, `NVDA`, `RELIANCE.NS`, `TCS.NS`, or `^NSEI`.

The notebook writes downloaded raw data under `data/` and analysis exports under `outputs/`.

## Run the Streamlit dashboard

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

The app opens in your browser, usually at http://localhost:8501.

1. Enter a ticker such as `AAPL` or `RELIANCE.NS`.
2. Choose start and end dates.
3. Review price, moving average, volume, returns, volatility, and drawdown charts.
4. Open **Model evaluation** to compare Random Forest with the naive baseline.
5. Download analyzed data and test predictions as CSV.

## Dataset source and manual download

The app/notebook fetches public historical data using the `yfinance` Python package. Search for a symbol at https://finance.yahoo.com/, open its **Historical Data** section, choose the period, and use the download option if available.

To load a manually downloaded CSV in the notebook, adapt the loading cell:

```python
raw = pd.read_csv("data/your_download.csv", parse_dates=["Date"], index_col="Date")
```

Yahoo Finance availability, symbol coverage, download behavior, and terms may change. This package is an open-source interface to Yahoo Finance data, not an official Yahoo Finance API.

## How the machine learning evaluation works

- Features are calculated using information available at date `t`.
- The target is the closing price at the next observed trading session (`t+1`).
- Rows are kept in chronological order.
- The first 80% of usable observations are training data; the last 20% are holdout data.
- The Random Forest model is compared with a naive baseline that predicts the next close as the current close.
- Metrics: MAE, RMSE, and R².

A lower MAE/RMSE is generally better. A model that does not beat the naive baseline on the holdout set has not demonstrated useful predictive improvement. For serious research, add walk-forward validation, compare against multiple baselines, consider adjusted prices and corporate actions carefully, and account for transaction costs and slippage before evaluating any trading strategy.

## Troubleshooting

### No data returned
- Check the ticker spelling and date range.
- Try `AAPL` or `MSFT`.
- Check your internet connection and retry later.
- Try the manual CSV download route.

### `ModuleNotFoundError`
Activate the same virtual environment where dependencies were installed, then run:

```bash
python -m pip install -r requirements.txt
```

### `streamlit` is not recognized
Use:

```bash
python -m streamlit run app.py
```

### Notebook uses the wrong Python environment
Install and register a kernel from the activated environment:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name stock-analysis --display-name "Python (stock-analysis)"
```

Then select **Python (stock-analysis)** as the notebook kernel.

### Empty charts or insufficient data
Use a longer date range and a valid ticker. Some indicators require 20–50 trading sessions and the model requires enough observations after feature engineering.

### Yahoo Finance request errors
Wait and retry, reduce repeated downloads, or use a manually downloaded CSV. The dashboard caches successful data temporarily.

## Suggested GitHub upload

Create a new GitHub repository named `stock-market-data-analysis`. Do not commit your virtual environment. If you add API keys or other credentials later, keep them out of GitHub.

```bash
git init
git add app.py requirements.txt README.md notebooks/
git commit -m "Build stock market data analysis project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/stock-market-data-analysis.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Add `data/` and `outputs/` only if you intentionally want to share the generated files.

## Resume bullet (edit to match your actual results)

**Stock Market Data Analysis | Python, Pandas, yfinance, Scikit-learn, Streamlit**
- Analyzed historical stock prices and trading volume; engineered returns, moving averages, volatility, and drawdown indicators.
- Built an interactive Streamlit dashboard and evaluated a Random Forest next-session price baseline against a naive forecast using chronological holdout metrics.

Only report model performance or business impact after you have run the project and verified the results.
