import pandas as pd


def align_news_to_trading_days(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Map each article to the same or next available trading day for its stock."""
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


def aggregate_daily_sentiment(aligned_news: pd.DataFrame) -> pd.DataFrame:
    return (
        aligned_news.dropna(subset=["trading_day"])
        .groupby(["stock", "trading_day"], as_index=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "size"),
        )
    )


def prepare_price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values(["stock", "Date"])
    df["daily_return_pct"] = df.groupby("stock")["Adj Close"].pct_change() * 100
    return df[["stock", "Date", "daily_return_pct"]].dropna()


def sentiment_return_dataset(news_sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    aligned = align_news_to_trading_days(news_sentiment, prices)
    daily_sentiment = aggregate_daily_sentiment(aligned)
    returns = prepare_price_returns(prices)
    merged = daily_sentiment.merge(
        returns,
        left_on=["stock", "trading_day"],
        right_on=["stock", "Date"],
        how="inner",
    )
    return merged.drop(columns=["Date"])


def pearson_correlation(dataset: pd.DataFrame) -> float:
    if len(dataset) < 2:
        return float("nan")
    return float(dataset["avg_sentiment"].corr(dataset["daily_return_pct"], method="pearson"))

