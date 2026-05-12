import pandas as pd
from textblob import TextBlob


def textblob_polarity(text: str) -> float:
    """
    Compute TextBlob polarity score for a given text.
    Parameters:
        text (str): Input text.
    Returns:
        float: Polarity score in [-1, 1].
    """
    return float(TextBlob(str(text)).sentiment.polarity)


def classify_sentiment(score: float, threshold: float = 0.05) -> str:
    """
    Classify sentiment score as positive, negative, or neutral.
    Parameters:
        score (float): Sentiment score.
        threshold (float): Neutral threshold (default 0.05).
    Returns:
        str: Sentiment label.
    """
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def add_sentiment_scores(news: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment scores and labels to news DataFrame using TextBlob.
    Parameters:
        news (pd.DataFrame): News DataFrame with 'headline'.
    Returns:
        pd.DataFrame: DataFrame with 'sentiment_score' and 'sentiment_label'.
    """
    df = news.copy()
    try:
        df["sentiment_score"] = df["headline"].map(textblob_polarity)
        df["sentiment_label"] = df["sentiment_score"].map(classify_sentiment)
    except Exception as e:
        raise RuntimeError(f"Failed to compute sentiment scores: {e}") from e
    return df

