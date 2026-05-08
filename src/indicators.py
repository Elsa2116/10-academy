import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }
    )


def add_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    close = df["Close"]
    df["sma_20"] = sma(close, 20)
    df["sma_50"] = sma(close, 50)
    df["ema_20"] = ema(close, 20)
    df["rsi_14"] = rsi(close, 14)
    return pd.concat([df, macd(close)], axis=1)


def add_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["daily_return_pct"] = df["Adj Close"].pct_change() * 100
    return df

