"""
Context processors para FinancialRAG.
Provee variables globales a todas las plantillas (base.html y vistas hijas).
"""
from core.models import StockPrediction


def pipeline_status(request):
    """Retorna información del estado del pipeline activa para la barra lateral."""
    try:
        sources_count = StockPrediction.objects.values('ticker').distinct().count()
        latest_pred = StockPrediction.objects.order_by('-date').values_list('date', flat=True).first()
        updated_str = latest_pred.strftime('%b %d, %Y') if latest_pred else 'recently'
    except Exception:
        sources_count = 0
        updated_str = 'offline'

    return {
        'pipeline_sources': sources_count,
        'pipeline_updated': updated_str,
    }
