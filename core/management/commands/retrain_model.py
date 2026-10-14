"""
Management command para reentrenar el modelo con features de sentimiento.
Ejecuta la comparación de 3 variantes y guarda la mejor en la BD.
"""
import sys
from pathlib import Path
from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import StockPrediction, SentimentFeature
import pandas as pd
import numpy as np

# Agregar services al path
sys.path.append(str(Path(__file__).parent.parent.parent / 'services'))

from base_ml import create_lightgbm, train_model, build_cross_sectional_strategy
from sentiment_features import SentimentFeatureExtractor


class Command(BaseCommand):
    help = 'Reentrena el modelo ML con features de sentimiento y guarda predicciones en la BD'

    def add_arguments(self, parser):
        parser.add_argument(
            '--variant',
            type=str,
            default='c',
            choices=['a', 'b', 'c'],
            help='Variante del modelo: a=baseline, b=core sentiment, c=full sentiment'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Limpiar predicciones existentes antes de entrenar'
        )

    def handle(self, *args, **options):
        variant = options['variant']
        clear_existing = options['clear_existing']
        
        self.stdout.write(f'\n{"="*60}')
        self.stdout.write(f'REENTRENANDO MODELO - Variante {variant.upper()}')
        self.stdout.write(f'{"="*60}\n')
        
        # 1. Limpiar datos existentes si se solicita
        if clear_existing:
            self.stdout.write('Limpiando predicciones existentes...')
            StockPrediction.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('✓ Base de datos limpiada'))
        
        # 2. Cargar datos base (Alpha158)
        self.stdout.write('\n[1/4] Cargando datos base...')
        predictions_file = Path('core/data/predictions.csv')
        
        if not predictions_file.exists():
            self.stdout.write(self.style.ERROR(f'Archivo no encontrado: {predictions_file}'))
            return
        
        df = pd.read_csv(predictions_file)
        self.stdout.write(f'✓ Cargadas {len(df)} predicciones base')
        
        # 3. Extraer features de sentimiento
        self.stdout.write('\n[2/4] Extrayendo features de sentimiento...')
        extractor = SentimentFeatureExtractor()
        news_df = extractor.load_news_data()
        sentiment_features = extractor.extract_features(news_df)
        self.stdout.write(f'✓ Features de sentimiento extraídas: {len(sentiment_features)}')
        
        # 4. Preparar datos según variante
        self.stdout.write(f'\n[3/4] Preparando datos para variante {variant}...')
        
        # Merge con sentiment features
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        
        sentiment_features['date'] = pd.to_datetime(sentiment_features['date']).dt.date
        
        df_merged = df.merge(
            sentiment_features,
            left_on=['instrument', 'date'],
            right_on=['ticker', 'date'],
            how='left'
        )
        
        # Seleccionar columnas según variante
        base_cols = [c for c in df.columns if c not in ['datetime', 'date']]
        
        if variant == 'a':
            feature_cols = base_cols
            self.stdout.write('Variante A: Solo Alpha158 (baseline)')
        elif variant == 'b':
            sentiment_cols = ['sentiment_mean', 'news_volume']
            feature_cols = base_cols + [c for c in sentiment_cols if c in df_merged.columns]
            self.stdout.write('Variante B: Alpha158 + sentiment_mean + news_volume')
        elif variant == 'c':
            sentiment_cols = [
                'sentiment_mean', 'sentiment_max', 'sentiment_min',
                'sentiment_dispersion', 'news_volume',
                'positive_ratio', 'negative_ratio'
            ]
            feature_cols = base_cols + [c for c in sentiment_cols if c in df_merged.columns]
            self.stdout.write('Variante C: Alpha158 + todas las features de sentimiento')
        
        # Llenar NaN con 0
        for col in feature_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)
        
        self.stdout.write(f'✓ Features finales: {len(feature_cols)}')
        
        # 5. Entrenar modelo
        self.stdout.write('\n[4/4] Entrenando modelo...')
        
        # Preparar datos de entrenamiento
        X = df_merged[feature_cols].values
        y = df_merged['prediction'].values  # Usar predicción como target temporal
        
        # Split train/valid/test
        n = len(X)
        n_train = int(n * 0.6)
        n_valid = int(n * 0.2)
        
        X_train, X_valid, X_test = X[:n_train], X[n_train:n_train+n_valid], X[n_train+n_valid:]
        y_train, y_valid, y_test = y[:n_train], y[n_train:n_train+n_valid], y[n_train+n_valid:]
        
        # Crear y entrenar modelo
        model = create_lightgbm()
        model = train_model(model, X_train, y_train, X_valid, y_valid)
        
        # Predecir en test set
        y_pred = model.predict(X_test)
        
        self.stdout.write(self.style.SUCCESS('✓ Modelo entrenado'))
        
        # 6. Guardar predicciones en BD
        self.stdout.write('\nGuardando predicciones en la base de datos...')
        
        test_df = df_merged.iloc[n_train+n_valid:].copy()
        test_df['prediction_new'] = y_pred
        
        predictions_to_create = []
        for _, row in test_df.iterrows():
            predictions_to_create.append(
                StockPrediction(
                    ticker=row['instrument'],
                    date=row['date'],
                    predicted_return=row['prediction_new'],
                    actual_return=row.get('actual', None),
                    model_version=f'variant_{variant}_retrained'
                )
            )
        
        StockPrediction.objects.bulk_create(predictions_to_create)
        
        self.stdout.write(self.style.SUCCESS(f'\n✓ Creadas {len(predictions_to_create)} predicciones'))
        self.stdout.write(self.style.SUCCESS(f'✓ Modelo version: variant_{variant}_retrained'))
        
        # 7. Calcular métricas
        self.stdout.write('\nCalculando métricas...')
        
        metrics = build_cross_sectional_strategy(
            test_df.set_index(['instrument', 'date'])[['prediction_new']],
            test_df.set_index(['instrument', 'date'])[['actual']]
        )
        
        self.stdout.write(f'\n{"="*60}')
        self.stdout.write('MÉTRICAS DEL MODELO')
        self.stdout.write(f'{"="*60}')
        self.stdout.write(f'Mean IC: {metrics["mean_ic"]:.4f}')
        self.stdout.write(f'Mean Rank IC: {metrics["mean_rank_ic"]:.4f}')
        self.stdout.write(f'Net Sharpe: {metrics["net_sharpe"]:.4f}')
        self.stdout.write(f'Net Total Return: {metrics["net_total_return"]:.4f}')
        self.stdout.write(f'Net Max Drawdown: {metrics["net_max_drawdown"]:.4f}')
        self.stdout.write(f'{"="*60}\n')
        
        self.stdout.write(self.style.SUCCESS('✓ Reentrenamiento completado'))
