import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def headline_length_stats(news: pd.DataFrame) -> pd.Series:
    lengths = news["headline"].fillna("").str.len()
    return lengths.describe()
        """
        Compute descriptive statistics for headline lengths.
        Parameters:
            news (pd.DataFrame): News DataFrame with 'headline'.
        Returns:
            pd.Series: Descriptive statistics for headline length.
        """
        lengths = news["headline"].fillna("").str.len()
        return lengths.describe()


def add_headline_length(news: pd.DataFrame) -> pd.DataFrame:
    df = news.copy()
    df["headline_length"] = df["headline"].fillna("").str.len()
    return df
        """
        Add a 'headline_length' column to the news DataFrame.
        Parameters:
            news (pd.DataFrame): News DataFrame with 'headline'.
        Returns:
            pd.DataFrame: DataFrame with 'headline_length' column.
        """
        df = news.copy()
        df["headline_length"] = df["headline"].fillna("").str.len()
        return df


def top_publishers(news: pd.DataFrame, n: int = 15) -> pd.Series:
    return news["publisher"].fillna("Unknown").value_counts().head(n)
        """
        Return the top-n publishers by article count.
        Parameters:
            news (pd.DataFrame): News DataFrame with 'publisher'.
            n (int): Number of top publishers to return.
        Returns:
            pd.Series: Publisher counts.
        """
        return news["publisher"].fillna("Unknown").value_counts().head(n)


def publication_frequency(news: pd.DataFrame, freq: str = "D") -> pd.Series:
    return news.set_index("date").sort_index().resample(freq).size()
        """
        Compute publication frequency by date.
        Parameters:
            news (pd.DataFrame): News DataFrame with 'date'.
            freq (str): Resample frequency (default 'D' for daily).
        Returns:
            pd.Series: Article count per period.
        """
        return news.set_index("date").sort_index().resample(freq).size()


def news_volume_spikes(news: pd.DataFrame, freq: str = "D", z_threshold: float = 2.0) -> pd.DataFrame:
    """Identify unusually high publication-volume periods with z-scores."""
    volume = publication_frequency(news, freq)
    std = volume.std(ddof=0)
    if std == 0 or pd.isna(std):
        z_score = pd.Series(0.0, index=volume.index)
    else:
        z_score = (volume - volume.mean()) / std
    return (
        pd.DataFrame(
            {
                "period": volume.index,
                "article_count": volume.values,
                "z_score": z_score.values,
            }
        )
        .query("z_score >= @z_threshold")
        .sort_values(["z_score", "article_count"], ascending=False)
        .reset_index(drop=True)
    )
        """
        Identify unusually high publication-volume periods with z-scores.
        Parameters:
            news (pd.DataFrame): News DataFrame.
            freq (str): Resample frequency.
            z_threshold (float): Z-score threshold for spike detection.
        Returns:
            pd.DataFrame: DataFrame of periods with high publication volume.
        """
        volume = publication_frequency(news, freq)
        std = volume.std(ddof=0)
        if std == 0 or pd.isna(std):
            z_score = pd.Series(0.0, index=volume.index)
        else:
            z_score = (volume - volume.mean()) / std
        return (
            pd.DataFrame(
                {
                    "period": volume.index,
                    "article_count": volume.values,
                    "z_score": z_score.values,
                }
            )
            .query("z_score >= @z_threshold")
            .sort_values(["z_score", "article_count"], ascending=False)
            .reset_index(drop=True)
        )


def publishing_hour_distribution(news: pd.DataFrame) -> pd.Series:
    return news["date"].dt.hour.value_counts().sort_index()
        """
        Compute distribution of article publication hours.
        Parameters:
            news (pd.DataFrame): News DataFrame with 'date'.
        Returns:
            pd.Series: Article count by hour of day.
        """
        return news["date"].dt.hour.value_counts().sort_index()


def extract_publisher_domain(publisher: str) -> str:
    if "@" not in publisher:
        return "unknown"
    return publisher.split("@")[-1].strip().lower()
        """
        Extract domain from publisher string if present.
        Parameters:
            publisher (str): Publisher string.
        Returns:
            str: Extracted domain or 'unknown'.
        """
        if "@" not in publisher:
            return "unknown"
        return publisher.split("@")[-1].strip().lower()


def publisher_domains(news: pd.DataFrame) -> pd.Series:
    return news["publisher"].fillna("").map(extract_publisher_domain).value_counts()
        """
        Count publisher domains in the news DataFrame.
        Parameters:
            news (pd.DataFrame): News DataFrame with 'publisher'.
        Returns:
            pd.Series: Domain counts.
        """
        return news["publisher"].fillna("").map(extract_publisher_domain).value_counts()


def publisher_coverage_summary(news: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Summarize publisher volume, stock breadth, and active date range."""
    return (
        news.assign(date_only=news["date"].dt.date)
        .groupby("publisher")
        .agg(
            article_count=("headline", "size"),
            unique_stocks=("stock", "nunique"),
            first_date=("date_only", "min"),
            last_date=("date_only", "max"),
        )
        .sort_values("article_count", ascending=False)
        .head(n)
        .reset_index()
    )


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


def headline_theme_counts(news: pd.DataFrame) -> pd.DataFrame:
    """Count interpretable finance themes from headline keywords."""
    patterns = {
        "earnings/results": r"\b(?:earnings|eps|revenue|results|profit|loss|guidance)\b",
        "analyst ratings": r"\b(?:upgrade|downgrade|price target|rating|initiates|maintains)\b",
        "price movement": r"\b(?:gains|falls|jumps|drops|rises|slumps|surges|trades higher|trades lower)\b",
        "corporate actions": r"\b(?:merger|acquisition|buyback|dividend|split|offering)\b",
        "regulatory/legal": r"\b(?:fda|approval|sec|lawsuit|probe|regulatory|patent)\b",
    }
    headlines = news["headline"].fillna("").str.lower()
    rows = [
        {
            "theme": theme,
            "article_count": int(headlines.str.contains(pattern, regex=True).sum()),
        }
        for theme, pattern in patterns.items()
    ]
    return pd.DataFrame(rows).sort_values("article_count", ascending=False).reset_index(drop=True)

