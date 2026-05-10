import numpy as np
import pandas as pd


DEFAULT_INDICATOR_PARAMS = {
    "sma_windows": (20, 50),
    "ema_span": 20,
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}


def _load_talib():
    try:
        import talib
    except ImportError:
        return None
    return talib


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


def add_talib_indicators(
    prices: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Add SMA, EMA, RSI, and MACD using TA-Lib.

    Default parameter choices follow common technical-analysis conventions:
    20/50-session SMA for short/medium trend, 20-session EMA for responsive
    trend, 14-session RSI with 30/70 interpretation bands, and 12/26/9 MACD.
    """
    talib = _load_talib()
    if talib is None:
        raise ImportError("TA-Lib is not installed. Install the TA-Lib package to use this engine.")

    settings = DEFAULT_INDICATOR_PARAMS | (params or {})
    df = prices.copy()
    close = df["Close"].astype(float)

    for window in settings["sma_windows"]:
        df[f"sma_{window}"] = talib.SMA(close, timeperiod=window)
    df[f"ema_{settings['ema_span']}"] = talib.EMA(close, timeperiod=settings["ema_span"])
    df[f"rsi_{settings['rsi_window']}"] = talib.RSI(close, timeperiod=settings["rsi_window"])

    macd_line, signal_line, histogram = talib.MACD(
        close,
        fastperiod=settings["macd_fast"],
        slowperiod=settings["macd_slow"],
        signalperiod=settings["macd_signal"],
    )
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    df.attrs["indicator_engine"] = "TA-Lib"
    df.attrs["indicator_params"] = settings
    return df


def add_pandas_indicators(
    prices: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Add the same indicators with pandas formulas as a reproducible fallback."""
    settings = DEFAULT_INDICATOR_PARAMS | (params or {})
    df = prices.copy()
    close = df["Close"]
    for window in settings["sma_windows"]:
        df[f"sma_{window}"] = sma(close, window)
    df[f"ema_{settings['ema_span']}"] = ema(close, settings["ema_span"])
    df[f"rsi_{settings['rsi_window']}"] = rsi(close, settings["rsi_window"])
    df = pd.concat(
        [
            df,
            macd(
                close,
                fast=settings["macd_fast"],
                slow=settings["macd_slow"],
                signal=settings["macd_signal"],
            ),
        ],
        axis=1,
    )
    df.attrs["indicator_engine"] = "pandas fallback"
    df.attrs["indicator_params"] = settings
    return df


def add_technical_indicators(
    prices: pd.DataFrame,
    use_talib: bool = True,
    params: dict | None = None,
) -> pd.DataFrame:
    """Add Task 2 indicators, preferring TA-Lib and falling back to pandas.

    Set ``use_talib=False`` to force the transparent pandas implementation for
    debugging or environments where TA-Lib cannot be installed.
    """
    if use_talib:
        try:
            return add_talib_indicators(prices, params=params)
        except ImportError:
            pass
    return add_pandas_indicators(prices, params=params)


def add_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["daily_return_pct"] = df["Adj Close"].pct_change() * 100
    return df

