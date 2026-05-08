import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


def save_current_figure(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")


def plot_news_volume(freq):
    fig, ax = plt.subplots(figsize=(12, 4))
    freq.plot(ax=ax)
    ax.set_title("News Publication Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Article count")
    return fig, ax


def plot_top_publishers(counts):
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Most Active Publishers")
    ax.set_xlabel("Article count")
    ax.set_ylabel("")
    return fig, ax


def plot_price_indicators(df, ticker="Stock"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], label="Close", linewidth=1.4)
    for col in ["sma_20", "sma_50", "ema_20"]:
        if col in df:
            ax.plot(df["Date"], df[col], label=col.upper(), linewidth=1)
    ax.set_title(f"{ticker} Price With Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig, ax


def plot_sentiment_return_scatter(df, correlation):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="avg_sentiment", y="daily_return_pct", hue="stock", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Daily Sentiment vs Return (Pearson r={correlation:.3f})")
    ax.set_xlabel("Average daily sentiment")
    ax.set_ylabel("Daily return (%)")
    return fig, ax

