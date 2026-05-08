import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def headline_length_stats(news: pd.DataFrame) -> pd.Series:
    lengths = news["headline"].fillna("").str.len()
    return lengths.describe()


def add_headline_length(news: pd.DataFrame) -> pd.DataFrame:
    df = news.copy()
    df["headline_length"] = df["headline"].fillna("").str.len()
    return df


def top_publishers(news: pd.DataFrame, n: int = 15) -> pd.Series:
    return news["publisher"].fillna("Unknown").value_counts().head(n)


def publication_frequency(news: pd.DataFrame, freq: str = "D") -> pd.Series:
    return news.set_index("date").sort_index().resample(freq).size()


def publishing_hour_distribution(news: pd.DataFrame) -> pd.Series:
    return news["date"].dt.hour.value_counts().sort_index()


def extract_publisher_domain(publisher: str) -> str:
    if "@" not in publisher:
        return "unknown"
    return publisher.split("@")[-1].strip().lower()


def publisher_domains(news: pd.DataFrame) -> pd.Series:
    return news["publisher"].fillna("").map(extract_publisher_domain).value_counts()


def top_keywords(news: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(news["headline"].fillna(""))
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    return (
        pd.DataFrame({"term": terms, "score": scores})
        .sort_values("score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def recurring_phrases(news: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(2, 3),
        min_df=2,
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(news["headline"].fillna(""))
    counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    return (
        pd.DataFrame({"phrase": terms, "count": counts})
        .sort_values("count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

