import pandas as pd

from src.indicators import add_daily_returns, ema, rsi, sma


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

