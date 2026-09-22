from django.shortcuts import render
from django.http import JsonResponse
from core.services import (
    get_latest_predictions,
    calculate_prediction_metrics,
    get_latest_sentiment,
)


def home(request):
    """Página principal con resumen de predicciones y sentimiento."""
    predictions = get_latest_predictions(limit=6)
    sentiment = get_latest_sentiment(limit=6)
    
    # Combinar predicciones con sentimiento
    stocks_data = []
    for pred in predictions:
        sent = next((s for s in sentiment if s['ticker'] == pred['ticker']), None)
        
        # Determinar sentimiento basado en score
        if sent and sent['sentiment_mean'] > 0.2:
            sentiment_label = 'Bullish'
            sentiment_bg = '#071f0e'
            sentiment_color = '#6ee7b7'
        elif sent and sent['sentiment_mean'] < -0.2:
            sentiment_label = 'Bearish'
            sentiment_bg = '#3f1111'
            sentiment_color = '#fca5a5'
        else:
            sentiment_label = 'Neutral'
            sentiment_bg = '#1c1917'
            sentiment_color = '#a8a29e'
        
        stocks_data.append({
            'ticker': pred['ticker'],
            'predicted_return': pred['predicted_return'],
            'date': pred['date'],
            'sentiment': sentiment_label,
            'sentiment_bg': sentiment_bg,
            'sentiment_color': sentiment_color,
        })
    
    return render(request, 'core/home.html', {'stocks': stocks_data})


def portfolio(request):
    """Portafolio con predicciones reales del modelo."""
    from core.models import StockPrediction, SentimentFeature
    from django.db.models import Avg, Count
    from datetime import datetime
    
    # Obtener métricas reales del portafolio
    latest_date = StockPrediction.objects.order_by('-date').values_list('date', flat=True).first()
    
    if latest_date:
        # Predicciones del último día (primero filtrar, luego slice)
        predictions_qs = StockPrediction.objects.filter(date=latest_date).order_by('ticker')
        predictions_with_actual = predictions_qs.filter(actual_return__isnull=False)
        predictions = predictions_qs[:50]
        
        # Calcular métricas
        total_positions = predictions_qs.count()
        total_predictions = StockPrediction.objects.count()
        mean_predicted = predictions.aggregate(avg=Avg('predicted_return'))['avg'] or 0
        
        # Calcular correlación con actual_return
        if predictions_with_actual.count() > 1:
            import pandas as pd
            pred_values = list(predictions_with_actual.values_list('predicted_return', flat=True))
            actual_values = list(predictions_with_actual.values_list('actual_return', flat=True))
            correlation = pd.Series(pred_values).corr(pd.Series(actual_values))
        else:
            correlation = 0
        
        # Preparar datos para el template
        stocks = []
        for pred in predictions:
            # Buscar sentimiento para este ticker
            sent = SentimentFeature.objects.filter(
                ticker=pred.ticker,
                date__lte=pred.date
            ).order_by('-date').first()
            
            # Determinar sentimiento
            if sent and sent.sentiment_mean > 0.2:
                sentiment_label = 'Bullish'
                sentiment_bg = '#071f0e'
                sentiment_color = '#6ee7b7'
                sentiment_border = '#064e3b'
            elif sent and sent.sentiment_mean < -0.2:
                sentiment_label = 'Bearish'
                sentiment_bg = '#3f1111'
                sentiment_color = '#fca5a5'
                sentiment_border = '#7f1d1d'
            else:
                sentiment_label = 'Neutral'
                sentiment_bg = '#1c1917'
                sentiment_color = '#a8a29e'
                sentiment_border = '#44403c'
            
            stocks.append({
                'ticker': pred.ticker,
                'date': pred.date,
                'predicted_return': pred.predicted_return,
                'actual_return': pred.actual_return,
                'sentiment': sentiment_label,
                'sentiment_bg': sentiment_bg,
                'sentiment_color': sentiment_color,
                'sentiment_border': sentiment_border,
            })
        
        context = {
            'stocks': stocks,
            'total_positions': total_positions,
            'total_predictions': total_predictions,
            'mean_predicted': mean_predicted,
            'correlation': correlation,
            'last_updated': latest_date.strftime('%b %d, %Y · %H:%M EST'),
        }
    else:
        context = {
            'stocks': [],
            'total_positions': 0,
            'total_predictions': 0,
            'mean_predicted': 0,
            'correlation': 0,
            'last_updated': 'N/A',
        }
    
    return render(request, 'core/portfolio.html', context)


def newsfeed(request):
    """Feed de noticias con análisis de sentimiento."""
    from core.models import NewsArticle
    
    articles = list(
        NewsArticle.objects
        .order_by('-date', 'ticker')[:10]
        .values('id', 'ticker', 'date', 'headline', 'source', 'sentiment_score')
    )
    
    return render(request, 'core/newsfeed.html', {'articles': articles})


def analytics(request):
    """Analíticas del modelo."""
    metrics = calculate_prediction_metrics()
    
    context = {
        'metrics': metrics,
    }
    
    return render(request, 'core/analytics.html', context)


def settings(request):
    """Configuración de fuentes de noticias."""
    all_sources = [
        {'name': 'Reuters', 'active': True},
        {'name': 'Bloomberg', 'active': True},
        {'name': 'WSJ', 'active': True},
        {'name': 'CNBC', 'active': True},
        {'name': 'FT', 'active': False},
        {'name': "Barron's", 'active': False},
        {'name': 'Seeking Alpha', 'active': False},
        {'name': 'Motley Fool', 'active': False},
    ]
    return render(request, 'core/settings.html', {'all_sources': all_sources})


# API endpoints para datos ML
def api_predictions(request, ticker=None):
    """API endpoint para obtener predicciones."""
    if ticker:
        from core.services import get_predictions_for_ticker
        data = get_predictions_for_ticker(ticker)
    else:
        data = get_latest_predictions(limit=50)
    
    return JsonResponse({'predictions': data})


def api_sentiment(request, ticker=None):
    """API endpoint para obtener sentimiento."""
    if ticker:
        from core.services import get_sentiment_for_ticker
        data = get_sentiment_for_ticker(ticker)
    else:
        data = get_latest_sentiment(limit=50)
    
    return JsonResponse({'sentiment': data})


def api_metrics(request):
    """API endpoint para métricas del modelo."""
    metrics = calculate_prediction_metrics()
    return JsonResponse({'metrics': metrics or {}})
