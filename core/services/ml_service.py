"""
Servicios ML para FinancialRAG.

Este módulo proporciona funciones para:
- Cargar y ejecutar el modelo de predicción
- Obtener predicciones para tickers específicos
- Calcular métricas de rendimiento
"""

from pathlib import Path
import pandas as pd
from django.conf import settings
from core.models import StockPrediction, SentimentFeature


# Ruta al archivo de predicciones
PREDICTIONS_FILE = Path(settings.BASE_DIR) / 'core' / 'data' / 'predictions.csv'


def load_predictions_from_csv():
    """Carga predicciones desde CSV y las guarda en la BD."""
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"No se encontró {PREDICTIONS_FILE}")
    
    df = pd.read_csv(PREDICTIONS_FILE)
    
    # El CSV tiene columnas: datetime, instrument, prediction, actual
    predictions_to_create = []
    for _, row in df.iterrows():
        predictions_to_create.append(
            StockPrediction(
                ticker=row['instrument'],
                date=pd.to_datetime(row['datetime']).date(),
                predicted_return=row['prediction'],
                actual_return=row.get('actual'),
                model_version='alpha158_baseline'
            )
        )
    
    # Bulk create para eficiencia
    StockPrediction.objects.bulk_create(predictions_to_create, ignore_conflicts=True)
    return len(predictions_to_create)


def get_predictions_for_ticker(ticker, start_date=None, end_date=None):
    """Obtiene predicciones para un ticker específico."""
    queryset = StockPrediction.objects.filter(ticker=ticker)
    
    if start_date:
        queryset = queryset.filter(date__gte=start_date)
    if end_date:
        queryset = queryset.filter(date__lte=end_date)
    
    return list(queryset.values('date', 'predicted_return', 'actual_return'))


def get_latest_predictions(limit=10):
    """Obtiene las predicciones más recientes."""
    return list(
        StockPrediction.objects
        .order_by('-date', 'ticker')
        .values('ticker', 'date', 'predicted_return', 'actual_return')[:limit]
    )


def calculate_prediction_metrics(ticker=None, model_version=None):
    """Calcula métricas completas de rendimiento del modelo (Accuracy, RMSE, MAE, R², IC)."""
    import numpy as np
    
    if not model_version:
        if StockPrediction.objects.filter(model_version='variant_c_sentiment_augmented').exists():
            model_version = 'variant_c_sentiment_augmented'
        else:
            model_version = 'alpha158_baseline'
            
    queryset = StockPrediction.objects.filter(
        model_version=model_version,
        actual_return__isnull=False
    )
    
    if ticker:
        queryset = queryset.filter(ticker=ticker)
    
    predictions = list(queryset.values('predicted_return', 'actual_return'))
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    y_pred = df['predicted_return'].values
    y_true = df['actual_return'].values
    
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    correlation = float(pd.Series(y_pred).corr(pd.Series(y_true)))
    
    # Directional Accuracy (Hit Rate / Tasa de acierto de dirección)
    correct_direction = (np.sign(y_pred) == np.sign(y_true))
    accuracy = float(np.mean(correct_direction) * 100)
    
    return {
        'count': len(predictions),
        'accuracy': round(accuracy, 2),
        'correlation': round(correlation, 4),
        'ic': round(correlation, 4),
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'mean_predicted': float(np.mean(y_pred)),
        'mean_actual': float(np.mean(y_true)),
        'model_version': model_version,
        'baseline_accuracy': 50.04,
        'baseline_ic': 0.0091,
    }

