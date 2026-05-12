from pathlib import Path

import pandas as pd


NEWS_COLUMNS = {"headline", "url", "publisher", "date", "stock"}
PRICE_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}


def load_news(path: str | Path) -> pd.DataFrame:
    """
    Load and lightly normalize the financial news dataset.

    Parameters:
        path (str | Path): Path to the CSV file containing news data.
    Returns:
        pd.DataFrame: Cleaned news DataFrame with required columns and types.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"News file not found at {path}. Please check the path and try again.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read news CSV: {e}") from e

    missing = NEWS_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"News dataset is missing columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["headline"] = df["headline"].fillna("").astype(str)
    df["publisher"] = df["publisher"].fillna("Unknown").astype(str)
    df["stock"] = df["stock"].fillna("").astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date", "stock"])
    if df.empty:
        raise ValueError("No valid news records found after cleaning. Check input data for required fields and formats.")
    return df


def load_price_data(path: str | Path) -> pd.DataFrame:
    """
    Load historical OHLCV data and normalize numeric/date columns.

    Parameters:
        path (str | Path): Path to the CSV file containing price data.
    Returns:
        pd.DataFrame: Cleaned price DataFrame with required columns and types.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Price file not found at {path}. Please check the path and try again.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read price CSV: {e}") from e
    return normalize_price_data(df)


def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize historical OHLCV data from CSV files or yfinance downloads.

    Parameters:
        df (pd.DataFrame): Raw price DataFrame.
    Returns:
        pd.DataFrame: Cleaned and normalized price DataFrame.
    Raises:
        ValueError: If required columns are missing or no valid records remain after cleaning.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if str(col[0]).lower() in {c.lower() for c in PRICE_COLUMNS} else col[-1]
            for col in df.columns
        ]

    if "Date" not in df.columns and df.index.name in {"Date", "Datetime"}:
        df = df.reset_index()

    missing = PRICE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Price dataset is missing columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "Adj Close"])
    if df.empty:
        raise ValueError("No valid price records found after cleaning. Check input data for required fields and formats.")
    return df.sort_values("Date").reset_index(drop=True)

