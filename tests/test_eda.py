import pandas as pd

from src.eda import headline_theme_counts, news_volume_spikes, publisher_coverage_summary


def test_news_volume_spikes_flags_high_volume_day():
    news = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-02", "2026-01-03"]
            ),
            "headline": ["a", "b", "c", "d", "e"],
            "publisher": ["p"] * 5,
            "stock": ["AAPL"] * 5,
        }
    )
    spikes = news_volume_spikes(news, z_threshold=1.0)
    assert pd.Timestamp("2026-01-02") in set(spikes["period"])


def test_publisher_coverage_summary_counts_stock_breadth():
    news = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "headline": ["a", "b", "c"],
            "publisher": ["Reuters", "Reuters", "Other"],
            "stock": ["AAPL", "MSFT", "AAPL"],
        }
    )
    summary = publisher_coverage_summary(news, n=1)
    assert summary.loc[0, "publisher"] == "Reuters"
    assert summary.loc[0, "unique_stocks"] == 2


def test_headline_theme_counts_finds_business_topics():
    news = pd.DataFrame(
        {
            "headline": [
                "Apple earnings beat revenue estimates",
                "Analyst upgrades Tesla price target",
                "Biotech jumps after FDA approval",
            ]
        }
    )
    themes = headline_theme_counts(news).set_index("theme")
    assert themes.loc["earnings/results", "article_count"] == 1
    assert themes.loc["analyst ratings", "article_count"] == 1
    assert themes.loc["regulatory/legal", "article_count"] == 1
