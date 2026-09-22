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
    """Portafolio con predicciones del modelo."""
    predictions = get_latest_predictions(limit=10)
    
    stocks = []
    for pred in predictions:
        stocks.append({
            'ticker': pred['ticker'],
            'predicted_return': f"{pred['predicted_return']:.2%}",
            'date': pred['date'],
        })
    
    return render(request, 'core/portfolio.html', {'stocks': stocks})


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
