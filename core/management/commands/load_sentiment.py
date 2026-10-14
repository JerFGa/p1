from django.core.management.base import BaseCommand
from core.models import NewsArticle, SentimentFeature
import pandas as pd
from pathlib import Path
import numpy as np


class Command(BaseCommand):
    help = 'Carga noticias y genera features de sentimiento'

    def add_arguments(self, parser):
        parser.add_argument(
            '--news-file',
            type=str,
            default='data/financial_news.csv',
            help='Ruta al archivo CSV de noticias'
        )
        parser.add_argument(
            '--generate-synthetic',
            action='store_true',
            help='Generar noticias sintéticas si no existe el archivo'
        )

    def handle(self, *args, **options):
        news_file = Path(options['news_file'])
        
        # Limpiar datos existentes
        NewsArticle.objects.all().delete()
        SentimentFeature.objects.all().delete()
        self.stdout.write('Base de datos limpiada')
        
        # Cargar o generar noticias
        if news_file.exists():
            self.stdout.write(f'Cargando noticias desde {news_file}...')
            df = pd.read_csv(news_file)
        elif options['generate_synthetic']:
            self.stdout.write('Generando noticias sintéticas...')
            df = self._generate_synthetic_news()
        else:
            self.stdout.write(
                self.style.WARNING(
                    f'Archivo no encontrado: {news_file}\n'
                    f'Usa --generate-synthetic para generar datos de prueba'
                )
            )
            return
        
        self.stdout.write(f'Leídas {len(df)} noticias')
        
        # Crear artículos en bulk
        articles = []
        for _, row in df.iterrows():
            articles.append(
                NewsArticle(
                    ticker=row['ticker'],
                    date=row['date'],
                    headline=row.get('headline', ''),
                    text=row.get('text', ''),
                    sentiment_score=row.get('sentiment_score', None)
                )
            )
        
        NewsArticle.objects.bulk_create(articles)
        self.stdout.write(
            self.style.SUCCESS(f'Creados {len(articles)} artículos')
        )
        
        # Generar features de sentimiento
        self.stdout.write('Generando features de sentimiento...')
        self._generate_sentiment_features(df)
        
    def _generate_synthetic_news(self):
        """Genera noticias sintéticas para testing"""
        np.random.seed(42)
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        dates = pd.date_range(start='2017-01-01', end='2026-02-20', freq='B')
        
        data = []
        for date in dates:
            for ticker in tickers:
                # Generar 0-3 noticias por ticker por día
                n_news = np.random.poisson(1.5)
                for _ in range(n_news):
                    sentiment = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                    data.append({
                        'date': date,
                        'ticker': ticker,
                        'headline': f'Noticia para {ticker}',
                        'text': f'Texto de noticia financiera para {ticker}',
                        'sentiment_score': sentiment
                    })
        
        return pd.DataFrame(data)
    
    def _generate_sentiment_features(self, news_df):
        """Genera features de sentimiento agrupadas por ticker y fecha"""
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Agrupar por ticker y fecha
        grouped = news_df.groupby(['ticker', 'date']).agg(
            sentiment_mean=('sentiment_score', 'mean'),
            sentiment_max=('sentiment_score', 'max'),
            sentiment_min=('sentiment_score', 'min'),
            sentiment_dispersion=('sentiment_score', 'std'),
            news_volume=('sentiment_score', 'count')
        ).reset_index()
        
        # Llenar NaN en dispersión con 0
        grouped['sentiment_dispersion'] = grouped['sentiment_dispersion'].fillna(0)
        
        # Crear features en bulk
        features = []
        for _, row in grouped.iterrows():
            features.append(
                SentimentFeature(
                    ticker=row['ticker'],
                    date=row['date'],
                    sentiment_mean=row['sentiment_mean'],
                    sentiment_max=row['sentiment_max'],
                    sentiment_dispersion=row['sentiment_dispersion'],
                    news_volume=int(row['news_volume'])
                )
            )
        
        SentimentFeature.objects.bulk_create(features)
        self.stdout.write(
            self.style.SUCCESS(f'Creadas {len(features)} features de sentimiento')
        )
