import json
from datetime import timedelta

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg

from core.models import StockPrediction, SentimentFeature, NewsArticle
from core.services import (
    get_latest_predictions,
    calculate_prediction_metrics,
    get_latest_sentiment,
    get_predictions_for_ticker,
    get_sentiment_for_ticker,
    execute_rag_query,
)
from core.views.pages import get_current_model_version


@csrf_exempt
def api_chat(request):
    """Endpoint RAG conversacional: recupera contexto y retorna análisis fundamentado con citas."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        message = data.get('message', '').strip()
    except Exception:
        message = request.POST.get('message', '').strip()

    if not message:
        return JsonResponse({'error': 'El mensaje no puede estar vacío'}, status=400)

    session_key = request.session.session_key or ''
    result = execute_rag_query(message, session_key=session_key)

    return JsonResponse({
        'status': 'ok',
        'text': result['text'],
        'citations': result['citations'],
        'ticker': result['ticker'],
        'latency_ms': result['latency_ms'],
    })


@csrf_exempt
def api_save_settings(request):
    """Guarda las preferencias del usuario en su sesión actual."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        current_settings = request.session.get('user_settings', {})
        current_settings.update(data)
        request.session['user_settings'] = current_settings
        request.session.modified = True
        return JsonResponse({'status': 'ok', 'settings': current_settings})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def api_analytics_volume(request):
    """Endpoint para actualizar dinámicamente la gráfica según rango temporal."""
    range_param = request.GET.get('range', '14d')
    days_map = {'7d': 7, '14d': 14, '30d': 30, '90d': 90}
    days = days_map.get(range_param, 14)

    version = get_current_model_version()
    latest_date = StockPrediction.objects.filter(model_version=version).order_by('-date').values_list('date', flat=True).first()
    if not latest_date:
        return JsonResponse({'data': []})

    start_date = latest_date - timedelta(days=days)
    daily_returns = list(
        StockPrediction.objects.filter(model_version=version, date__gte=start_date)
        .values('date')
        .annotate(avg_return=Avg('predicted_return'))
        .order_by('date')
    )

    data = [
        {
            'date': c['date'].strftime('%b %d'),
            'return_pct': round((c['avg_return'] or 0) * 100, 3)
        }
        for c in daily_returns
    ]
    return JsonResponse({'data': data, 'range': range_param})


def api_ticker_history(request, ticker):
    """Retorna la serie temporal histórica de predicciones y sentimiento para un ticker específico."""
    ticker = ticker.upper().strip()
    version = get_current_model_version()

    history = list(
        StockPrediction.objects.filter(model_version=version, ticker=ticker)
        .order_by('-date')[:30]
        .values('date', 'predicted_return', 'actual_return')
    )
    history.reverse()

    sentiments = list(
        SentimentFeature.objects.filter(ticker=ticker)
        .order_by('-date')[:30]
        .values('date', 'sentiment_mean', 'news_volume')
    )
    sent_map = {s['date']: s for s in sentiments}

    labels = []
    pred_vals = []
    actual_vals = []
    sentiment_vals = []

    for item in history:
        d = item['date']
        labels.append(d.strftime('%b %d'))
        pred_vals.append(round(item['predicted_return'] * 100, 2))
        actual_vals.append(round(item['actual_return'] * 100, 2) if item['actual_return'] is not None else None)
        sent = sent_map.get(d)
        sentiment_vals.append(round(sent['sentiment_mean'], 2) if sent else 0.0)

    articles = list(
        NewsArticle.objects.filter(ticker=ticker).order_by('-date')[:3].values('headline', 'source', 'date', 'sentiment_score')
    )

    return JsonResponse({
        'ticker': ticker,
        'labels': labels,
        'predicted_returns': pred_vals,
        'actual_returns': actual_vals,
        'sentiments': sentiment_vals,
        'articles': [
            {
                'headline': a['headline'],
                'source': a['source'],
                'date': a['date'].strftime('%b %d, %Y') if hasattr(a['date'], 'strftime') else str(a['date']),
                'sentiment_score': a['sentiment_score']
            }
            for a in articles
        ]
    })


def api_predictions(request, ticker=None):
    """API endpoint para obtener predicciones."""
    if ticker:
        data = get_predictions_for_ticker(ticker)
    else:
        data = get_latest_predictions(limit=50)

    return JsonResponse({'predictions': data})


def api_sentiment(request, ticker=None):
    """API endpoint para obtener sentimiento."""
    if ticker:
        data = get_sentiment_for_ticker(ticker)
    else:
        data = get_latest_sentiment(limit=50)

    return JsonResponse({'sentiment': data})


def api_metrics(request):
    """API endpoint para métricas del modelo."""
    metrics = calculate_prediction_metrics()
    return JsonResponse({'metrics': metrics or {}})
