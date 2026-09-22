"""
Módulo para extraer features de sentimiento de noticias financieras.
Usa FinBERT para analizar sentimiento de headlines y textos de noticias.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import pipeline
from tqdm import tqdm


class SentimentFeatureExtractor:
    """Extrae features de sentimiento desde noticias financieras."""
    
    def __init__(self, cache_dir=None):
        """
        Inicializa el extractor de sentimiento.
        
        Args:
            cache_dir: Directorio para cachear resultados de FinBERT
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/sentiment_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar FinBERT
        print("Cargando modelo FinBERT...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1  # CPU
        )
        print("FinBERT cargado")
    
    def load_news_data(self, news_file=None):
        """
        Carga dataset de noticias financieras.
        
        Args:
            news_file: Path al archivo CSV de noticias
            
        Returns:
            DataFrame con columnas: date, ticker, headline, text
        """
        if news_file is None:
            news_file = Path('data/financial_news.csv')
        
        if not Path(news_file).exists():
            print(f"Archivo {news_file} no encontrado. Generando datos sintéticos...")
            return self._generate_synthetic_news()
        
        df = pd.read_csv(news_file, parse_dates=['date'])
        print(f"Loaded {len(df)} noticias desde {news_file}")
        return df
    
    def _generate_synthetic_news(self, start_date='2017-01-01', end_date='2026-02-20'):
        """
        Genera datos sintéticos de noticias para testing.
        
        Returns:
            DataFrame con noticias sintéticas
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        data = []
        for date in dates:
            for ticker in tickers:
                # Generar 0-3 noticias por ticker por día
                n_news = np.random.poisson(1.5)
                for _ in range(n_news):
                    data.append({
                        'date': date,
                        'ticker': ticker,
                        'headline': f'Noticia sintética para {ticker}',
                        'text': f'Texto sintético de noticia financiera para {ticker} en {date.date()}'
                    })
        
        df = pd.DataFrame(data)
        print(f"Generadas {len(df)} noticias sintéticas")
        return df
    
    def compute_sentiment(self, text):
        """
        Calcula sentimiento de un texto usando FinBERT.
        
        Args:
            text: Texto a analizar
            
        Returns:
            dict con sentiment_score (-1 a 1) y sentiment_label
        """
        if not text or pd.isna(text):
            return {'sentiment_score': 0.0, 'sentiment_label': 'neutral'}
        
        try:
            result = self.sentiment_pipeline(str(text)[:512])[0]
            label = result['label'].lower()
            score = result['score']
            
            # Convertir a score continuo (-1 a 1)
            if label == 'positive':
                sentiment_score = score
            elif label == 'negative':
                sentiment_score = -score
            else:  # neutral
                sentiment_score = 0.0
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': label
            }
        except Exception as e:
            print(f"Error computando sentimiento: {e}")
            return {'sentiment_score': 0.0, 'sentiment_label': 'neutral'}
    
    def extract_features(self, news_df, use_cache=True):
        """
        Extrae features de sentimiento por (ticker, date).
        
        Args:
            news_df: DataFrame con noticias
            use_cache: Si debe usar caché de resultados
            
        Returns:
            DataFrame con features de sentimiento por (ticker, date)
        """
        cache_file = self.cache_dir / 'sentiment_features.parquet'
        
        if use_cache and cache_file.exists():
            print(f"Cargando features de sentimiento desde caché: {cache_file}")
            return pd.read_parquet(cache_file)
        
        print("Extrayendo features de sentimiento...")
        
        # Analizar sentimiento de cada noticia
        sentiments = []
        for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Analizando sentimiento"):
            text = row.get('text', row.get('headline', ''))
            sentiment = self.compute_sentiment(text)
            sentiments.append({
                'date': row['date'],
                'ticker': row['ticker'],
                'sentiment_score': sentiment['sentiment_score'],
                'sentiment_label': sentiment['sentiment_label']
            })
        
        sentiment_df = pd.DataFrame(sentiments)
        
        # Agregar por (ticker, date)
        features = sentiment_df.groupby(['ticker', 'date']).agg(
            sentiment_mean=('sentiment_score', 'mean'),
            sentiment_max=('sentiment_score', lambda x: x.max() if len(x) > 0 else 0),
            sentiment_min=('sentiment_score', lambda x: x.min() if len(x) > 0 else 0),
            sentiment_std=('sentiment_score', 'std'),
            news_volume=('sentiment_score', 'count'),
            positive_ratio=('sentiment_label', lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0),
            negative_ratio=('sentiment_label', lambda x: (x == 'negative').sum() / len(x) if len(x) > 0 else 0)
        ).reset_index()
        
        # Llenar NaN en std con 0 (cuando solo hay 1 noticia)
        features['sentiment_std'] = features['sentiment_std'].fillna(0)
        
        # Renombrar columnas para consistencia
        features = features.rename(columns={
            'sentiment_std': 'sentiment_dispersion'
        })
        
        print(f"Features extraídas: {len(features)} combinaciones (ticker, date)")
        
        # Guardar en caché
        if use_cache:
            features.to_parquet(cache_file, index=False)
            print(f"Features guardadas en caché: {cache_file}")
        
        return features
    
    def merge_with_predictions(self, predictions_df, sentiment_features_df):
        """
        Mergea features de sentimiento con predicciones existentes.
        
        Args:
            predictions_df: DataFrame con predicciones (datetime, instrument, prediction)
            sentiment_features_df: DataFrame con features de sentimiento
            
        Returns:
            DataFrame mergeado
        """
        # Asegurar tipos compatibles
        predictions_df = predictions_df.copy()
        predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
        predictions_df['date'] = predictions_df['datetime'].dt.date
        
        sentiment_features_df = sentiment_features_df.copy()
        sentiment_features_df['date'] = pd.to_datetime(sentiment_features_df['date']).dt.date
        
        # Merge
        merged = predictions_df.merge(
            sentiment_features_df,
            left_on=['instrument', 'date'],
            right_on=['ticker', 'date'],
            how='left'
        )
        
        # Llenar NaN con 0 (días sin noticias)
        sentiment_cols = ['sentiment_mean', 'sentiment_max', 'sentiment_min', 
                         'sentiment_dispersion', 'news_volume', 
                         'positive_ratio', 'negative_ratio']
        
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        # Drop columnas redundantes
        merged = merged.drop(columns=['ticker', 'date'], errors='ignore')
        
        print(f"Merged DataFrame: {len(merged)} filas, {len(merged.columns)} columnas")
        
        return merged


if __name__ == '__main__':
    # Testing
    extractor = SentimentFeatureExtractor()
    news_df = extractor.load_news_data()
    features = extractor.extract_features(news_df)
    print(features.head())
    print(f"\nColumnas: {features.columns.tolist()}")
