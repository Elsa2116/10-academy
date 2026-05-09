import pandas as pd

from src.data_loading import normalize_price_data
from src.indicators import add_daily_returns, add_technical_indicators, ema, macd, rsi, sma
from src.quant_metrics import add_financial_metrics, add_pynance_metrics


def test_sma_uses_window_mean():
    values = pd.Series([1, 2, 3, 4, 5])
    result = sma(values, 3)
    assert pd.isna(result.iloc[1])
    assert result.iloc[-1] == 4


def test_ema_keeps_series_length():
    values = pd.Series([1, 2, 3, 4, 5])
    assert len(ema(values, 3)) == len(values)


def test_rsi_bounds_after_warmup():
    values = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8, 7, 9, 10, 11, 12, 13])
    result = rsi(values, 14).dropna()
    assert ((result >= 0) & (result <= 100)).all()


def test_daily_returns_percent_change():
    prices = pd.DataFrame({"Adj Close": [100, 110, 99]})
    result = add_daily_returns(prices)
    assert round(result.loc[1, "daily_return_pct"], 2) == 10.0
    assert round(result.loc[2, "daily_return_pct"], 2) == -10.0


def test_macd_returns_expected_columns():
    values = pd.Series(range(1, 40))
    result = macd(values)
    assert list(result.columns) == ["macd", "macd_signal", "macd_hist"]
    assert len(result) == len(values)


def test_add_technical_indicators_includes_task_two_columns():
    prices = pd.DataFrame(
        {
            "Close": range(1, 61),
            "Adj Close": range(1, 61),
        }
    )
    result = add_technical_indicators(prices)
    expected = {"sma_20", "sma_50", "ema_20", "rsi_14", "macd", "macd_signal", "macd_hist"}
    assert expected.issubset(result.columns)


def test_normalize_price_data_parses_dates_and_numbers():
    raw = pd.DataFrame(
        {
            "Date": ["2026-01-02"],
            "Open": ["100"],
            "High": ["110"],
            "Low": ["95"],
            "Close": ["105"],
            "Adj Close": ["104"],
            "Volume": ["1000000"],
        }
    )
    result = normalize_price_data(raw)
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result.loc[0, "Close"] == 105


def test_financial_metrics_add_risk_return_columns():
    prices = pd.DataFrame({"Adj Close": [100, 102, 101, 105, 104]})
    result = add_financial_metrics(prices)
    expected = {"return", "cumulative_return", "rolling_volatility_20", "rolling_sharpe_20", "drawdown"}
    assert expected.issubset(result.columns)
    assert result.loc[0, "drawdown"] == 0


def test_pynance_metrics_returns_status_and_metrics():
    prices = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=30),
            "Adj Close": range(100, 130),
        }
    )
    result = add_pynance_metrics(prices)
    assert "pynance_status" in result.attrs
    assert {"pynance_return", "pynance_risk_20"}.issubset(result.columns) or {
        "return",
        "rolling_volatility_20",
    }.issubset(result.columns)

