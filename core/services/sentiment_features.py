"""
Sentiment Features Module for FinancialRAG

Extends the Qlib Alpha158 + LightGBM pipeline with news sentiment features.
Uses FinBERT (ProsusAI/finbert) to infer sentiment from financial news.
"""

from pathlib import Path
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "core" / "data"
CACHE_DIR = DATA_DIR / "sentiment_cache"

# Date range (must match lgb_sp500_qlb.py)
DATA_START = "2008-01-01"
DATA_END = "2026-02-20"

# FinBERT model
FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment feature names
SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_max",
    "sentiment_dispersion",
    "news_volume",
]


# ============================================================
# LOAD NEWS DATASET
# ============================================================

def load_news_dataset():
    """
    Load financial news dataset with timestamps and tickers.
    
    Expected format: CSV with columns [date, ticker, headline, text]
    If FNSPID is not available, uses synthetic news for demonstration.
    """
    print("\n" + "=" * 70)
    print("LOADING NEWS DATASET")
    print("=" * 70)
    
    news_file = DATA_DIR / "financial_news.csv"
    
    if news_file.exists():
        print(f"\nLoading news from: {news_file}")
        news = pd.read_csv(news_file)
        news["date"] = pd.to_datetime(news["date"])
        print(f"Loaded {len(news):,} news articles")
        return news
    
    print("\nNo news dataset found. Creating synthetic news for demonstration...")
    print("To use real data, place 'financial_news.csv' in core/data/ with columns:")
    print("  [date, ticker, headline, text]")
    
    # Synthetic news for demonstration
    news = _create_synthetic_news()
    return news


def _create_synthetic_news():
    """Create synthetic news data for demonstration."""
    np.random.seed(42)
    
    tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "GOOGL", "META", "JPM"]
    dates = pd.date_range(DATA_START, DATA_END, freq="B")
    
    news_list = []
    for date in dates:
        for ticker in tickers:
            if np.random.random() < 0.3:
                sentiment = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                news_list.append({
                    "date": date,
                    "ticker": ticker,
                    "headline": f"Sample news for {ticker}",
                    "text": f"Synthetic news text with sentiment {sentiment}",
                    "synthetic_sentiment": sentiment,
                })
    
    news = pd.DataFrame(news_list)
    print(f"Created {len(news):,} synthetic news articles")
    return news


# ============================================================
# COMPUTE SENTIMENT WITH FINBERT
# ============================================================

def compute_sentiment_with_finbert(news_df):
    """
    Compute sentiment scores using FinBERT.
    Caches results in parquet format for reproducibility.
    """
    print("\n" + "=" * 70)
    print("COMPUTING SENTIMENT WITH FINBERT")
    print("=" * 70)
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "finbert_sentiment.parquet"
    
    if cache_file.exists():
        print(f"\nLoading cached sentiment from: {cache_file}")
        sentiment_df = pd.read_parquet(cache_file)
        print(f"Loaded {len(sentiment_df):,} cached sentiment scores")
        return sentiment_df
    
    print("\nComputing sentiment with FinBERT...")
    print(f"Model: {FINBERT_MODEL}")
    
    try:
        from transformers import pipeline
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            truncation=True,
            max_length=512,
        )
        
        results = []
        for idx, row in news_df.iterrows():
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(news_df)}...")
            
            text = str(row.get("text", row.get("headline", "")))
            
            if "synthetic_sentiment" in row:
                score = row["synthetic_sentiment"]
            else:
                try:
                    result = sentiment_pipeline(text[:512])[0]
                    label = result["label"].lower()
                    score_val = result["score"]
                    
                    if label == "positive":
                        score = score_val
                    elif label == "negative":
                        score = -score_val
                    else:
                        score = 0.0
                except Exception as e:
                    print(f"    Error processing text: {e}")
                    score = 0.0
            
            results.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "sentiment_score": score,
            })
        
        sentiment_df = pd.DataFrame(results)
        sentiment_df.to_parquet(cache_file, index=False)
        print(f"\nSaved {len(sentiment_df):,} sentiment scores to cache")
        
    except ImportError:
        print("\nWARNING: transformers not available. Using synthetic sentiment.")
        sentiment_df = news_df[["date", "ticker", "synthetic_sentiment"]].copy()
        sentiment_df = sentiment_df.rename(columns={"synthetic_sentiment": "sentiment_score"})
    
    return sentiment_df


# ============================================================
# AGGREGATE SENTIMENT FEATURES
# ============================================================

def aggregate_sentiment_features(sentiment_df, instruments, date_range):
    """
    Aggregate sentiment scores by (instrument, date).
    
    Features computed:
    - sentiment_mean: Average sentiment score
    - sentiment_max: Maximum sentiment score
    - sentiment_dispersion: Standard deviation of sentiment
    - news_volume: Number of news articles
    """
    print("\n" + "=" * 70)
    print("AGGREGATING SENTIMENT FEATURES")
    print("=" * 70)
    
    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()
    
    grouped = sentiment_df.groupby(["ticker", "date"]).agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_max=("sentiment_score", lambda x: x.max() if len(x) > 0 else 0),
        sentiment_dispersion=("sentiment_score", "std"),
        news_volume=("sentiment_score", "count"),
    ).reset_index()
    
    grouped["sentiment_dispersion"] = grouped["sentiment_dispersion"].fillna(0.0)
    grouped = grouped.rename(columns={"ticker": "instrument"})
    
    print(f"\nAggregated sentiment for {len(grouped):,} (instrument, date) pairs")
    print(f"Instruments: {grouped['instrument'].nunique()}")
    print(f"Date range: {grouped['date'].min()} to {grouped['date'].max()}")
    
    return grouped


# ============================================================
# MERGE SENTIMENT WITH ALPHA158 FEATURES
# ============================================================

def merge_sentiment_with_features(X_df, sentiment_features_df):
    """
    Merge sentiment features with Alpha158 features.
    
    IMPORTANT: Only uses news with timestamp <= date of features (no lookahead).
    Fills missing sentiment with 0.0 (no news = neutral).
    """
    print("\n" + "=" * 70)
    print("MERGING SENTIMENT WITH ALPHA158 FEATURES")
    print("=" * 70)
    
    X_df = X_df.copy()
    sentiment_features_df = sentiment_features_df.copy()
    
    if isinstance(X_df.index, pd.MultiIndex):
        names = list(X_df.index.names)
        date_level = "datetime" if "datetime" in names else names[-1]
        inst_level = "instrument" if "instrument" in names else names[0]
        
        X_reset = X_df.reset_index()
        X_reset[date_level] = pd.to_datetime(X_reset[date_level]).dt.normalize()
        sentiment_features_df["date"] = pd.to_datetime(
            sentiment_features_df["date"]
        ).dt.normalize()
        
        merged = X_reset.merge(
            sentiment_features_df,
            left_on=[inst_level, date_level],
            right_on=["instrument", "date"],
            how="left",
        )
        
        for col in SENTIMENT_FEATURES:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)
        
        merged = merged.drop(columns=["date"], errors="ignore")
        merged = merged.set_index([inst_level, date_level])
        merged = merged.sort_index()
        
        print(f"\nMerged shape: {merged.shape}")
        print(f"Sentiment features added: {SENTIMENT_FEATURES}")
        print(f"Missing sentiment filled with 0.0")
        
        return merged
    
    else:
        print("\nWARNING: X_df does not have MultiIndex. Returning unchanged.")
        return X_df


# ============================================================
# MAIN FUNCTION
# ============================================================

def get_sentiment_features(instruments, date_range=None):
    """
    Main function to compute and return sentiment features.
    
    Returns:
        DataFrame with columns: [instrument, date, sentiment_mean, 
                                  sentiment_max, sentiment_dispersion, news_volume]
    """
    print("\n" + "=" * 70)
    print("SENTIMENT FEATURES PIPELINE")
    print("=" * 70)
    
    news_df = load_news_dataset()
    sentiment_df = compute_sentiment_with_finbert(news_df)
    sentiment_features = aggregate_sentiment_features(
        sentiment_df, instruments, date_range
    )
    
    return sentiment_features


if __name__ == "__main__":
    instruments = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"]
    sentiment_features = get_sentiment_features(instruments)
    print("\n" + "=" * 70)
    print("SENTIMENT FEATURES SAMPLE")
    print("=" * 70)
    print(sentiment_features.head(20))
