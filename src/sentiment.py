import pandas as pd
from textblob import TextBlob


def textblob_polarity(text: str) -> float:
    return float(TextBlob(str(text)).sentiment.polarity)


def classify_sentiment(score: float, threshold: float = 0.05) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def add_sentiment_scores(news: pd.DataFrame) -> pd.DataFrame:
    df = news.copy()
    df["sentiment_score"] = df["headline"].map(textblob_polarity)
    df["sentiment_label"] = df["sentiment_score"].map(classify_sentiment)
    return df

