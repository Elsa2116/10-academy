# Predicting Price Moves With News Sentiment

This repository contains a reproducible analysis pipeline for the Nova Financial
Solutions challenge. It connects financial news sentiment, technical indicators,
and daily stock returns to evaluate whether market narratives are associated
with short-term price movement.

## Project Structure

```text
.
├── .github/workflows/unittests.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_news.ipynb
│   ├── 02_quantitative_indicators.ipynb
│   └── 03_sentiment_correlation.ipynb
├── reports/
│   └── figures/
├── scripts/
├── src/
└── tests/
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name news-sentiment-analysis
```

## Data Placement

Add the FNSPID news CSV to `data/raw/`, for example:

```text
data/raw/fnspid_news.csv
```

Expected news columns:

```text
headline, url, publisher, date, stock
```

Add historical OHLCV stock price CSV files under `data/raw/prices/`.
Each price file should include:

```text
Date, Open, High, Low, Close, Adj Close, Volume
```

The notebooks also show how to download prices with `yfinance` if local price
CSV files are not available.

## Tasks Covered

Task 1:

- Descriptive headline statistics
- Publisher counts and publisher-domain extraction
- Publication volume by date and hour
- TF-IDF keyword and recurring phrase analysis

Task 2:

- Stock price cleaning
- SMA, EMA, RSI, MACD, and daily returns
- Visualizations for price, moving averages, RSI, and MACD

Task 3:

- Sentiment scoring using TextBlob polarity
- Weekend/holiday alignment to the next available trading day
- Daily sentiment aggregation by stock
- Pearson correlation between average daily sentiment and daily return
- Scatter and category-return visualizations

## Testing

```powershell
pytest -q
```

## Report Guidance

The final Medium-style report should include:

- Executive summary
- Methodology
- EDA insights
- Technical indicator findings
- Sentiment and correlation results
- Strategy recommendations
- Limitations and next steps

Keep the final report concise and limit it to no more than 10 plots.

