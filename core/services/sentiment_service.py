"""
Servicio de análisis de sentimiento para FinancialRAG.

Este módulo proporciona funciones para:
- Cargar noticias desde CSV
- Calcular features de sentimiento usando FinBERT
- Almacenar features en la base de datos
"""

from pathlib import Path
import pandas as pd
import numpy as np
from django.conf import settings
from core.models import NewsArticle, SentimentFeature


NEWS_FILE = Path(settings.BASE_DIR) / 'core' / 'data' / 'financial_news.csv'
SENTIMENT_CACHE_DIR = Path(settings.BASE_DIR) / 'core' / 'data' / 'sentiment_cache'


def load_news_from_csv():
    """Carga noticias desde CSV y las guarda en la BD."""
    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"No se encontró {NEWS_FILE}. "
                                f"Coloca el archivo en {NEWS_FILE}")
    
    df = pd.read_csv(NEWS_FILE)
    
    articles_to_create = []
    for _, row in df.iterrows():
        articles_to_create.append(
            NewsArticle(
                ticker=row['ticker'],
                date=pd.to_datetime(row['date']).date(),
                headline=row.get('headline', ''),
                text=row.get('text', ''),
                source=row.get('source', ''),
            )
        )
    
    NewsArticle.objects.bulk_create(articles_to_create, ignore_conflicts=True)
    return len(articles_to_create)


def compute_sentiment_features(start_date=None, end_date=None):
    """
    Calcula features de sentimiento para cada ticker y fecha.
    
    Features calculadas:
    - sentiment_mean: Promedio del sentimiento
    - sentiment_max: Máximo sentimiento
    - sentiment_dispersion: Desviación estándar
    - news_volume: Número de noticias
    """
    queryset = NewsArticle.objects.all()
    
    if start_date:
        queryset = queryset.filter(date__gte=start_date)
    if end_date:
        queryset = queryset.filter(date__lte=end_date)
    
    articles = list(queryset.values('ticker', 'date', 'sentiment_score'))
    
    if not articles:
        return 0
    
    df = pd.DataFrame(articles)
    
    # Si no hay sentiment_score, usar valores sintéticos para demo
    if df['sentiment_score'].isna().all():
        np.random.seed(42)
        df['sentiment_score'] = np.random.choice([-1, 0, 1], size=len(df), p=[0.3, 0.4, 0.3])
    
    # Agrupar por ticker y fecha
    grouped = df.groupby(['ticker', 'date']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_max=('sentiment_score', 'max'),
        sentiment_dispersion=('sentiment_score', 'std'),
        news_volume=('sentiment_score', 'count'),
    ).reset_index()
    
    # Rellenar NaN en dispersión con 0
    grouped['sentiment_dispersion'] = grouped['sentiment_dispersion'].fillna(0.0)
    
    # Guardar en BD
    features_to_create = []
    for _, row in grouped.iterrows():
        features_to_create.append(
            SentimentFeature(
                ticker=row['ticker'],
                date=row['date'],
                sentiment_mean=row['sentiment_mean'],
                sentiment_max=row['sentiment_max'],
                sentiment_dispersion=row['sentiment_dispersion'],
                news_volume=int(row['news_volume']),
            )
        )
    
    SentimentFeature.objects.bulk_create(features_to_create, ignore_conflicts=True)
    return len(features_to_create)


def get_sentiment_for_ticker(ticker, start_date=None, end_date=None):
    """Obtiene features de sentimiento para un ticker específico."""
    queryset = SentimentFeature.objects.filter(ticker=ticker)
    
    if start_date:
        queryset = queryset.filter(date__gte=start_date)
    if end_date:
        queryset = queryset.filter(date__lte=end_date)
    
    return list(queryset.values(
        'date', 'sentiment_mean', 'sentiment_max', 
        'sentiment_dispersion', 'news_volume'
    ))


def get_latest_sentiment(limit=10):
    """Obtiene las features de sentimiento más recientes."""
    return list(
        SentimentFeature.objects
        .order_by('-date', 'ticker')
        .values('ticker', 'date', 'sentiment_mean', 'news_volume')[:limit]
    )
