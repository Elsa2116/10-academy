import numpy as np
import pandas as pd


def maximum_drawdown(adj_close: pd.Series) -> pd.Series:
    """
    Return running drawdown from the cumulative adjusted-close peak.

    Parameters:
        adj_close (pd.Series): Adjusted close price series.
    Returns:
        pd.Series: Running drawdown values.
    """
    running_peak = adj_close.cummax()
    return (adj_close / running_peak) - 1


def add_financial_metrics(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Add return, volatility, Sharpe proxy, and drawdown metrics.

    Parameters:
        prices (pd.DataFrame): Price DataFrame with 'Adj Close'.
        periods_per_year (int): Number of trading periods per year (default 252).
    Returns:
        pd.DataFrame: DataFrame with added financial metrics.
    """
    df = prices.copy()
    returns = df["Adj Close"].pct_change()
    df["return"] = returns
    df["cumulative_return"] = (1 + returns.fillna(0)).cumprod() - 1
    df["rolling_volatility_20"] = returns.rolling(20, min_periods=20).std() * np.sqrt(periods_per_year)
    df["rolling_sharpe_20"] = (
        returns.rolling(20, min_periods=20).mean()
        / returns.rolling(20, min_periods=20).std()
        * np.sqrt(periods_per_year)
    )
    df["drawdown"] = maximum_drawdown(df["Adj Close"])
    return df


def add_pynance_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add PyNance return/risk metrics when the package imports cleanly.

    PyNance is included in the project requirements for the challenge, but some
    Python/package combinations can fail during import because of transitive
    dependency changes. In that case, the notebook still receives equivalent
    pandas-based metrics and records the reason in ``DataFrame.attrs``.

    Parameters:
        prices (pd.DataFrame): Price DataFrame with 'Adj Close'.
    Returns:
        pd.DataFrame: DataFrame with PyNance or fallback metrics.
    """
    df = prices.copy()
    try:
        from pynance.tech import movave, simple
    except Exception as exc:  # pragma: no cover - depends on local environment
        result = add_financial_metrics(df)
        result.attrs["pynance_status"] = f"fallback pandas metrics used: {exc}"
        return result

    try:
        indexed = df.set_index("Date") if "Date" in df.columns else df.copy()
        pynance_return = simple.ret(indexed, selection="Adj Close", outputcol="pynance_return")
        pynance_risk = movave.volatility(
            indexed,
            window=20,
            selection="Adj Close",
            outputcol="pynance_risk_20",
        )
        result = indexed.join([pynance_return, pynance_risk])
        if "Date" not in df.columns:
            result = result.reset_index(drop=True)
        return result
    except Exception as exc:
        fallback = add_financial_metrics(df)
        fallback.attrs["pynance_status"] = f"PyNance calculation failed, fallback used: {exc}"
        return fallback
    else:
        result = result.reset_index()
    result.attrs["pynance_status"] = "PyNance metrics computed"
    return result
