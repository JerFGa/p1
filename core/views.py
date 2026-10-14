import json
from datetime import datetime, timedelta
from collections import defaultdict

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Count, Q

from core.models import StockPrediction, SentimentFeature, NewsArticle, ChatQuery
from core.services import (
    get_latest_predictions,
    calculate_prediction_metrics,
    get_latest_sentiment,
    get_predictions_for_ticker,
    get_sentiment_for_ticker,
    execute_rag_query,
)

ACTIVE_MODEL_VERSION = 'variant_c_sentiment_augmented'


def get_current_model_version():
    """Retorna la versión activa del modelo (prioriza Variante C con Sentimiento)."""
    if StockPrediction.objects.filter(model_version=ACTIVE_MODEL_VERSION).exists():
        return ACTIVE_MODEL_VERSION
    return 'alpha158_baseline'


def home(request):
    """Página principal con resumen de mercado y asistente conversacional RAG."""
    if not request.session.session_key:
        request.session.save()

    version = get_current_model_version()
    latest_date = StockPrediction.objects.filter(model_version=version).order_by('-date').values_list('date', flat=True).first()

    # Obtener las predicciones más destacadas del día más reciente
    top_preds = list(
        StockPrediction.objects.filter(model_version=version, date=latest_date)
        .order_by('-predicted_return')[:6]
    )

    tickers = [p.ticker for p in top_preds]
    sentiments = {
        s.ticker: s for s in SentimentFeature.objects.filter(ticker__in=tickers, date__lte=latest_date).order_by('ticker', '-date')
    }

    stocks_data = []
    for pred in top_preds:
        sent = sentiments.get(pred.ticker)
        sent_mean = sent.sentiment_mean if sent else 0.0

        if sent_mean > 0.2:
            sentiment_label = 'Bullish'
            sentiment_bg = '#071f0e'
            sentiment_color = '#6ee7b7'
        elif sent_mean < -0.2:
            sentiment_label = 'Bearish'
            sentiment_bg = '#3f1111'
            sentiment_color = '#fca5a5'
        else:
            sentiment_label = 'Neutral'
            sentiment_bg = '#1c1917'
            sentiment_color = '#a8a29e'

        stocks_data.append({
            'ticker': pred.ticker,
            'predicted_return': pred.predicted_return * 100,
            'date': pred.date,
            'sentiment': sentiment_label,
            'sentiment_bg': sentiment_bg,
            'sentiment_color': sentiment_color,
        })

    unique_tickers = list(
        NewsArticle.objects.values_list('ticker', flat=True)
        .distinct()[:8]
    )
    if not unique_tickers:
        unique_tickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'JPM']

    categories = list(
        NewsArticle.objects.values_list('category', flat=True)
        .distinct()
        .order_by('category')
    )

    article_count = NewsArticle.objects.count()
    session_id = request.session.session_key[:8] if request.session.session_key else "New"

    context = {
        'stocks': stocks_data,
        'app_title': 'Financial News Assistant',
        'app_subtitle': 'Powered by RAG · Real-time market intelligence',
        'session_id': f'Session #{session_id}',
        'chat_placeholder': 'Ask about any stock, market news, or financial event…',
        'chat_instructions': 'Enter to send · Shift+Enter for new line',
        'welcome_message': "Hello! I'm FinancialRAG, your AI-powered financial assistant. Ask me about any S&P 500 company (e.g., AAPL, NVDA, TSLA), market trends, or macroeconomic events. I retrieve live predictions and news evidence with source citations.",
        'suggested_queries': [f'Analyze {ticker}' for ticker in unique_tickers[:5]],
        'categories': categories,
        'article_count': article_count,
        'active_model': 'Variant C (Alpha158 + FinBERT Sentiment)' if version == ACTIVE_MODEL_VERSION else 'Baseline Alpha158',
    }

    return render(request, 'core/home.html', context)


def portfolio(request):
    """Portafolio con métricas del modelo Alpha158 + LightGBM y análisis de rendimiento."""
    version = get_current_model_version()
    latest_date = StockPrediction.objects.filter(model_version=version).order_by('-date').values_list('date', flat=True).first()

    if latest_date:
        predictions_qs = StockPrediction.objects.filter(model_version=version, date=latest_date).order_by('ticker')
        total_positions = predictions_qs.count()
        total_predictions = StockPrediction.objects.filter(model_version=version).count()

        # Media de predicciones calculada sobre el total del día
        mean_predicted = predictions_qs.aggregate(avg=Avg('predicted_return'))['avg'] or 0

        # Correlación real con actual_return
        preds_with_actual = list(predictions_qs.filter(actual_return__isnull=False).values_list('predicted_return', 'actual_return'))
        if len(preds_with_actual) > 1:
            import pandas as pd
            df_corr = pd.DataFrame(preds_with_actual, columns=['p', 'a'])
            correlation = float(df_corr['p'].corr(df_corr['a']))
        else:
            correlation = 0.0

        # Listado para la tabla (primeras 60 posiciones o las más líquidas)
        predictions = list(predictions_qs[:60])
        tickers = [p.ticker for p in predictions]

        # Precargar sentimiento en 1 sola consulta
        sentiments_qs = SentimentFeature.objects.filter(
            ticker__in=tickers,
            date__lte=latest_date
        ).order_by('ticker', '-date')

        sent_map = {}
        for s in sentiments_qs:
            if s.ticker not in sent_map:
                sent_map[s.ticker] = s

        stocks = []
        for pred in predictions:
            sent = sent_map.get(pred.ticker)
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

            is_high_conviction = False
            if sent and abs(sent.sentiment_mean) >= 0.2:
                if (pred.predicted_return > 0 and sent.sentiment_mean > 0) or (pred.predicted_return < 0 and sent.sentiment_mean < 0):
                    if abs(pred.predicted_return) >= 0.03:
                        is_high_conviction = True

            stocks.append({
                'ticker': pred.ticker,
                'date': pred.date,
                'predicted_return': pred.predicted_return,
                'actual_return': pred.actual_return,
                'sentiment': sentiment_label,
                'sentiment_bg': sentiment_bg,
                'sentiment_color': sentiment_color,
                'sentiment_border': sentiment_border,
                'is_high_conviction': is_high_conviction,
            })

        # Retorno YTD anualizado
        ref_year = latest_date.year
        ytd_predictions = StockPrediction.objects.filter(
            model_version=version,
            date__year=ref_year
        ).aggregate(total=Avg('predicted_return'))['total'] or 0
        ytd_return = ytd_predictions * 100 * 252

        # Gráfico acumulativo de rendimiento (últimos 30 días de mercado)
        perf_by_date = StockPrediction.objects.filter(
            model_version=version,
            date__gte=latest_date - timedelta(days=45)
        ).values('date').annotate(
            avg_return=Avg('predicted_return')
        ).order_by('date')

        portfolio_value = 100000
        perf_chart_data = []
        for day_data in perf_by_date:
            portfolio_value *= (1 + (day_data['avg_return'] or 0))
            perf_chart_data.append({
                'label': day_data['date'].strftime('%b %d'),
                'value': round(portfolio_value, 2)
            })

        if not perf_chart_data:
            perf_chart_data = [{'label': latest_date.strftime('%b %d'), 'value': 100000}]

        context = {
            'stocks': stocks,
            'total_positions': total_positions,
            'total_predictions': total_predictions,
            'mean_predicted': mean_predicted * 100,
            'correlation': correlation,
            'last_updated': latest_date.strftime('%b %d, %Y'),
            'ytd_return': ytd_return,
            'perf_chart_json': json.dumps(perf_chart_data),
            'model_name': 'Variant C: Alpha158 + FinBERT' if version == ACTIVE_MODEL_VERSION else 'Baseline Alpha158',
        }
    else:
        context = {
            'stocks': [],
            'total_positions': 0,
            'total_predictions': 0,
            'mean_predicted': 0,
            'correlation': 0,
            'last_updated': 'N/A',
            'ytd_return': 0,
            'perf_chart_json': json.dumps([{'label': 'N/A', 'value': 100000}]),
            'model_name': 'N/A',
        }

    return render(request, 'core/portfolio.html', context)


def newsfeed(request):
    """Feed de noticias con análisis de sentimiento y resumen generado."""
    query = request.GET.get('q', '').strip()

    qs = NewsArticle.objects.all().order_by('-date', 'ticker')
    if query:
        qs = qs.filter(
            Q(headline__icontains=query) |
            Q(ticker__iexact=query) |
            Q(source__icontains=query) |
            Q(text__icontains=query)
        )

    raw_articles = list(
        qs[:60].values('id', 'ticker', 'date', 'headline', 'text', 'source', 'sentiment_score', 'category')
    )

    articles = []
    for art in raw_articles:
        text = art.get('text') or art.get('headline') or ''
        summary = text[:260] + "..." if len(text) > 260 else text

        score = art.get('sentiment_score')
        if score is not None and score > 0.2:
            sentiment_label = 'Bullish'
            sentiment_color = '#34d399'
            sentiment_bg = '#071f0e'
            sentiment_border = '#064e3b'
        elif score is not None and score < -0.2:
            sentiment_label = 'Bearish'
            sentiment_color = '#fca5a5'
            sentiment_bg = '#3f1111'
            sentiment_border = '#7f1d1d'
        else:
            sentiment_label = 'Neutral'
            sentiment_color = '#a8a29e'
            sentiment_bg = '#1c1917'
            sentiment_border = '#44403c'

        articles.append({
            'id': art['id'],
            'ticker': art['ticker'],
            'date': art['date'].strftime('%b %d, %Y') if hasattr(art['date'], 'strftime') else str(art['date']),
            'headline': art['headline'],
            'text': text,
            'summary': summary,
            'source': art['source'] or 'Verified Financial Wire',
            'category': art['category'] or 'General',
            'sentiment_label': sentiment_label,
            'sentiment_color': sentiment_color,
            'sentiment_bg': sentiment_bg,
            'sentiment_border': sentiment_border,
            'sentiment_score': round(score, 2) if score is not None else 0.0,
        })

    categories = list(
        NewsArticle.objects.values_list('category', flat=True)
        .distinct()
        .order_by('category')
    )

    context = {
        'articles': articles,
        'categories': categories,
        'article_count': NewsArticle.objects.count(),
        'search_query': query,
        'feed_subtitle': 'Real-time financial intelligence with FinBERT sentiment',
    }

    return render(request, 'core/newsfeed.html', context)


def analytics(request):
    """Analíticas del modelo, volatilidad real del mercado y telemetría RAG."""
    metrics = calculate_prediction_metrics()
    version = get_current_model_version()

    total_predictions = StockPrediction.objects.filter(model_version=version).count()
    unique_tickers = StockPrediction.objects.filter(model_version=version).values('ticker').distinct().count()
    total_articles = NewsArticle.objects.count()
    total_sentiment = SentimentFeature.objects.count()

    # Historial de consultas reales del asistente RAG
    recent_chat_queries = ChatQuery.objects.all().order_by('-created_at')[:12]
    queries_data = []
    for q in recent_chat_queries:
        queries_data.append({
            'text': q.query_text,
            'time': q.created_at.strftime('%H:%M · %b %d'),
            'latency': f"{q.latency_ms}ms",
            'sources': q.sources_count,
            'user': q.user_label or 'Analyst',
        })

    # Gráfica Dinámica: Retorno Promedio Diario del Mercado (en %)
    # Muestra la fluctuación real del S&P 500 día a día
    daily_returns = list(
        StockPrediction.objects.filter(model_version=version)
        .values('date')
        .annotate(avg_return=Avg('predicted_return'))
        .order_by('-date')[:14]
    )
    daily_returns.reverse()

    prediction_by_date_list = [
        {
            'date': p['date'].strftime('%b %d'),
            'return_pct': round((p['avg_return'] or 0) * 100, 3)
        }
        for p in daily_returns
    ]

    # Distribución real de fuentes de noticias (Multicolor)
    sources_qs = NewsArticle.objects.values('source').annotate(
        count=Count('id')
    ).order_by('-count')[:7]

    colors = ['#34d399', '#10b981', '#059669', '#3b82f6', '#6366f1', '#8b5cf6', '#ec4899']
    sources_list = []
    for idx, s in enumerate(sources_qs):
        source_name = s['source'] if s['source'] else 'Market Wire'
        sources_list.append({
            'name': source_name,
            'count': s['count'],
            'percentage': round(s['count'] / total_articles * 100, 1) if total_articles > 0 else 0,
            'color': colors[idx % len(colors)]
        })

    context = {
        'metrics': metrics,
        'total_predictions': total_predictions,
        'unique_tickers': unique_tickers,
        'total_articles': total_articles,
        'total_sentiment': total_sentiment,
        'prediction_by_date': json.dumps(prediction_by_date_list),
        'sources': sources_list,
        'sources_json': json.dumps([
            {'name': s['name'], 'value': s['count'], 'color': s['color']}
            for s in sources_list
        ]),
        'queries': queries_data,
        'active_model': 'Variant C (Alpha158 + FinBERT)' if version == ACTIVE_MODEL_VERSION else 'Baseline Alpha158',
    }

    return render(request, 'core/analytics.html', context)


def settings(request):
    """Configuración de fuentes de noticias y preferencias del usuario (con persistencia)."""
    db_sources = list(
        NewsArticle.objects.values_list('source', flat=True)
        .distinct()
        .order_by('source')
    )
    db_sources = [s for s in db_sources if s]
    if not db_sources:
        db_sources = ['Bloomberg', 'Reuters', 'The Wall Street Journal', 'CNBC', 'Financial Times', 'MarketWatch', "Barron's"]

    user_settings = request.session.get('user_settings', {
        'notif': True,
        'autoSumm': True,
        'darkMode': True,
        'active_sources': db_sources[:5],
    })

    active_set = set(user_settings.get('active_sources', db_sources[:5]))
    all_sources = [
        {'name': name, 'active': name in active_set}
        for name in db_sources
    ]

    context = {
        'all_sources': all_sources,
        'user_settings': user_settings,
        'active_source_count': len([s for s in all_sources if s['active']]),
    }
    return render(request, 'core/settings.html', context)


# ==============================================================================
# API Endpoints
# ==============================================================================

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
