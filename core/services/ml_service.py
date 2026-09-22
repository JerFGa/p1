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


def calculate_prediction_metrics(ticker=None):
    """Calcula métricas de rendimiento de las predicciones."""
    queryset = StockPrediction.objects.filter(actual_return__isnull=False)
    
    if ticker:
        queryset = queryset.filter(ticker=ticker)
    
    predictions = list(queryset.values('predicted_return', 'actual_return'))
    
    if not predictions:
        return None
    
    # Calcular correlación
    predicted = [p['predicted_return'] for p in predictions]
    actual = [p['actual_return'] for p in predictions]
    
    correlation = pd.Series(predicted).corr(pd.Series(actual))
    
    return {
        'count': len(predictions),
        'correlation': correlation,
        'mean_predicted': sum(predicted) / len(predicted),
        'mean_actual': sum(actual) / len(actual),
    }
