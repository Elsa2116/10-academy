import pandas as pd
import pytest

from src.correlation import (
    average_return_by_sentiment,
    interpret_correlation,
    pearson_correlation,
    sentiment_return_dataset,
)
from src.sentiment import classify_sentiment


def test_classify_sentiment_thresholds():
    assert classify_sentiment(0.2) == "positive"
    assert classify_sentiment(-0.2) == "negative"
    assert classify_sentiment(0.01) == "neutral"


def test_sentiment_return_dataset_aligns_weekend_to_next_trading_day():
    news = pd.DataFrame(
        {
            "stock": ["AAPL"],
            "date": [pd.Timestamp("2026-05-09 12:00:00", tz="UTC")],
            "headline": ["Apple shares rise after strong results"],
            "sentiment_score": [0.5],
        }
    )
    prices = pd.DataFrame(
        {
            "stock": ["AAPL", "AAPL"],
            "Date": ["2026-05-08", "2026-05-11"],
            "Adj Close": [100.0, 105.0],
        }
    )
    result = sentiment_return_dataset(news, prices)
    assert result.loc[0, "trading_day"] == pd.Timestamp("2026-05-11")
    assert result.loc[0, "daily_return_pct"] == pytest.approx(5.0)
    assert result.loc[0, "sentiment_category"] == "positive"


def test_pearson_correlation_returns_float():
    df = pd.DataFrame({"avg_sentiment": [-1, 0, 1], "daily_return_pct": [-2, 0, 2]})
    assert pearson_correlation(df) == 1.0


def test_average_return_by_sentiment_reindexes_categories():
    df = pd.DataFrame(
        {
            "avg_sentiment": [-0.2, 0.0, 0.3],
            "daily_return_pct": [-1.0, 0.2, 1.5],
        }
    )
    result = average_return_by_sentiment(df)
    assert list(result.index) == ["negative", "neutral", "positive"]
    assert result.loc["positive"] == 1.5


def test_interpret_correlation_describes_direction():
    assert "positive" in interpret_correlation(0.25)
    assert "negative" in interpret_correlation(-0.25)
