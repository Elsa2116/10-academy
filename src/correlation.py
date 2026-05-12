import pandas as pd

from src.sentiment import classify_sentiment


def align_news_to_trading_days(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Map each article to the same or next available trading day for its stock.

    Parameters:
        news (pd.DataFrame): News DataFrame with 'date' and 'stock'.
        prices (pd.DataFrame): Price DataFrame with 'Date' and 'stock'.
    Returns:
        pd.DataFrame: News DataFrame with 'trading_day' column.
    Raises:
        ValueError: If merge fails or required columns are missing.
    """
    try:
        news_df = news.copy()
        price_days = (
            prices[["stock", "Date"]]
            .drop_duplicates()
            .assign(Date=lambda df: pd.to_datetime(df["Date"]).dt.normalize())
            .sort_values(["stock", "Date"])
        )

        news_df["article_day"] = pd.to_datetime(news_df["date"], utc=True).dt.tz_convert(None).dt.normalize()
        news_df = news_df.sort_values(["stock", "article_day"])

        aligned = pd.merge_asof(
            news_df,
            price_days,
            left_on="article_day",
            right_on="Date",
            by="stock",
            direction="forward",
        )
        return aligned.rename(columns={"Date": "trading_day"})
    except Exception as e:
        raise RuntimeError(f"Failed to align news to trading days: {e}") from e


def aggregate_daily_sentiment(aligned_news: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sentiment for each stock and trading day.

    Parameters:
        aligned_news (pd.DataFrame): News DataFrame with 'trading_day'.
    Returns:
        pd.DataFrame: Aggregated sentiment and article count per stock-day.
    """
    return (
        aligned_news.dropna(subset=["trading_day"])
        .groupby(["stock", "trading_day"], as_index=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "size"),
        )
    )


def prepare_price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns for each stock.

    Parameters:
        prices (pd.DataFrame): Price DataFrame with 'Adj Close'.
    Returns:
        pd.DataFrame: DataFrame with daily return percentage.
    """
    df = prices.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values(["stock", "Date"])
    df["daily_return_pct"] = df.groupby("stock")["Adj Close"].pct_change() * 100
    return df[["stock", "Date", "daily_return_pct"]].dropna()


def sentiment_return_dataset(news_sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sentiment and returns for correlation analysis.

    Parameters:
        news_sentiment (pd.DataFrame): News DataFrame with sentiment scores.
        prices (pd.DataFrame): Price DataFrame.
    Returns:
        pd.DataFrame: DataFrame with sentiment, returns, and sentiment category.
    Raises:
        RuntimeError: If merge fails or results in empty DataFrame.
    """
    try:
        aligned = align_news_to_trading_days(news_sentiment, prices)
        daily_sentiment = aggregate_daily_sentiment(aligned)
        returns = prepare_price_returns(prices)
        merged = daily_sentiment.merge(
            returns,
            left_on=["stock", "trading_day"],
            right_on=["stock", "Date"],
            how="inner",
        )
        merged = merged.drop(columns=["Date"])
        merged["sentiment_category"] = merged["avg_sentiment"].map(classify_sentiment)
        if merged.empty:
            raise ValueError("No matched sentiment/return records after merging. Check data alignment and coverage.")
        return merged
    except Exception as e:
        raise RuntimeError(f"Failed to merge sentiment and returns: {e}") from e


def pearson_correlation(dataset: pd.DataFrame) -> float:
    if len(dataset) < 2:
        return float("nan")
    return float(dataset["avg_sentiment"].corr(dataset["daily_return_pct"], method="pearson"))


def average_return_by_sentiment(dataset: pd.DataFrame) -> pd.Series:
    """Return average daily return for negative, neutral, and positive days."""
    if "sentiment_category" not in dataset.columns:
        dataset = dataset.assign(
            sentiment_category=dataset["avg_sentiment"].map(classify_sentiment)
        )
    return (
        dataset.groupby("sentiment_category")["daily_return_pct"]
        .mean()
        .reindex(["negative", "neutral", "positive"])
    )


def interpret_correlation(correlation: float) -> str:
    """Create a concise plain-language interpretation of Pearson correlation."""
    if pd.isna(correlation):
        return "Not enough matched sentiment/return observations to estimate correlation."

    magnitude = abs(correlation)
    if magnitude < 0.1:
        strength = "very weak"
    elif magnitude < 0.3:
        strength = "weak"
    elif magnitude < 0.5:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "flat"
    return f"The same-day relationship is {strength} and {direction} (Pearson r={correlation:.3f})."

