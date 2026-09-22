"""
Motor RAG (Retrieval-Augmented Generation) para FinancialRAG.

Este servicio:
1. Extrae tickers o entidades del mensaje del usuario.
2. Recupera noticias relevantes (NewsArticle), predicciones ML (StockPrediction)
   y métricas de sentimiento (SentimentFeature).
3. Sintetiza un análisis financiero estructurado con citas referenciadas [1], [2].
4. Registra métricas de consulta (latencia, sesión, fuentes) en ChatQuery.
"""
import re
import time
from datetime import datetime
from django.db.models import Q
from core.models import StockPrediction, SentimentFeature, NewsArticle, ChatQuery

# Mapeo de nombres comunes de empresas a tickers
COMPANY_NAME_MAP = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'nvidia': 'NVDA',
    'meta': 'META',
    'facebook': 'META',
    'netflix': 'NFLX',
    'jpmorgan': 'JPM',
    'jp morgan': 'JPM',
    'bank of america': 'BAC',
    'visa': 'V',
    'walmart': 'WMT',
    'disney': 'DIS',
    'intel': 'INTC',
    'cisco': 'CSCO',
    'pfizer': 'PFE',
    'exxon': 'XOM',
    'chevron': 'CVX',
    'coca cola': 'KO',
    'coca-cola': 'KO',
    'pepsi': 'PEP',
    'oracle': 'ORCL',
    'adobe': 'ADBE',
    'broadcom': 'AVGO',
    'costco': 'COST',
    'amd': 'AMD',
}


def extract_tickers_from_query(query_text):
    """Extrae posibles tickers de la consulta del usuario."""
    q_lower = query_text.lower()
    matched_tickers = set()

    # 1. Búsqueda por nombres de empresas comunes
    for name, ticker in COMPANY_NAME_MAP.items():
        if re.search(r'\b' + re.escape(name) + r'\b', q_lower):
            matched_tickers.add(ticker)

    # 2. Búsqueda por formato de ticker directo (1 a 5 letras mayúsculas o minúsculas aisladas)
    words = re.findall(r'\b[a-zA-Z]{1,5}\b', query_text)
    for word in words:
        ticker_candidate = word.upper()
        # Verificar si existe en la base de datos de predicciones
        if StockPrediction.objects.filter(ticker=ticker_candidate).exists():
            matched_tickers.add(ticker_candidate)

    return list(matched_tickers)


def retrieve_context(ticker=None, query_text="", max_articles=4):
    """
    Recupera contexto de noticias y predicciones para el ticker o palabras clave.
    """
    context_data = {
        'ticker': ticker,
        'predictions': [],
        'sentiment': None,
        'articles': [],
    }

    if ticker:
        # Predicciones ML más recientes
        preds = StockPrediction.objects.filter(ticker=ticker).order_by('-date')[:5]
        context_data['predictions'] = list(preds.values('date', 'predicted_return', 'actual_return', 'model_version'))

        # Sentimiento más reciente
        sent = SentimentFeature.objects.filter(ticker=ticker).order_by('-date').first()
        if sent:
            context_data['sentiment'] = {
                'date': sent.date,
                'sentiment_mean': sent.sentiment_mean,
                'sentiment_max': sent.sentiment_max,
                'sentiment_dispersion': sent.sentiment_dispersion,
                'news_volume': sent.news_volume,
            }

        # Noticias específicas del ticker
        articles = NewsArticle.objects.filter(ticker=ticker).order_by('-date')[:max_articles]
        context_data['articles'] = list(articles)
    else:
        # Búsqueda temática por palabras clave en titulares o texto
        stopwords = {'what', 'when', 'where', 'which', 'about', 'how', 'tell', 'show', 'with', 'from', 'this', 'that', 'have', 'been'}
        keywords = [w for w in re.findall(r'\w+', query_text) if len(w) > 3 and w.lower() not in stopwords]
        
        articles = []
        if keywords:
            q_filter = Q()
            for kw in keywords[:5]:
                q_filter |= Q(headline__icontains=kw) | Q(text__icontains=kw)
            articles = list(NewsArticle.objects.filter(q_filter).order_by('-date')[:max_articles])
        
        if not articles:
            # Fallback a noticias macro o destacadas recientes
            articles = list(NewsArticle.objects.filter(category__in=['Macro & Fed', 'Markets']).order_by('-date')[:max_articles])
            if not articles:
                articles = list(NewsArticle.objects.order_by('-date')[:max_articles])

        context_data['articles'] = articles

    return context_data


def synthesize_rag_response(query_text, context_data):
    """
    Sintetiza la respuesta financiera integrando los datos cuantitativos y noticias con citas.
    """
    ticker = context_data.get('ticker')
    predictions = context_data.get('predictions', [])
    sentiment = context_data.get('sentiment')
    articles = context_data.get('articles', [])

    citations = []
    response_lines = []

    # Construir citas a partir de los artículos recuperados
    for idx, art in enumerate(articles, start=1):
        confidence = "High" if (art.sentiment_score is not None and abs(art.sentiment_score) > 0.3) else "Medium"
        snippet = art.text[:220] + "..." if len(art.text) > 220 else (art.text or art.headline)
        citations.append({
            'num': idx,
            'title': art.headline,
            'source': art.source or 'Financial Wire',
            'date': art.date.strftime('%b %d, %Y') if hasattr(art.date, 'strftime') else str(art.date),
            'snippet': snippet,
            'confidence': confidence,
            'text': art.text or art.headline,
        })

    if ticker:
        # Respuesta analítica para un ticker
        latest_pred = predictions[0] if predictions else None
        pred_return_pct = (latest_pred['predicted_return'] * 100) if latest_pred else 0.0
        pred_color = "#34d399" if pred_return_pct >= 0 else "#f87171"

        sent_mean = sentiment['sentiment_mean'] if sentiment else 0.0
        news_vol = sentiment['news_volume'] if sentiment else len(articles)

        if sent_mean > 0.2:
            sent_label = "Bullish (Positivo)"
            sent_color = "#34d399"
        elif sent_mean < -0.2:
            sent_label = "Bearish (Negativo)"
            sent_color = "#f87171"
        else:
            sent_label = "Neutral"
            sent_color = "#a8a29e"

        pred_date_str = latest_pred['date'].strftime('%b %d, %Y') if latest_pred and hasattr(latest_pred['date'], 'strftime') else (str(latest_pred['date']) if latest_pred else 'N/A')

        response_lines.append(f"<strong>Análisis RAG para {ticker}</strong> (Datos actualizados al {pred_date_str}):<br><br>")
        response_lines.append(f"• <strong>Retorno Predicho (LightGBM Alpha158)</strong>: <span style='color: {pred_color}; font-weight: bold;'>{pred_return_pct:+.2f}%</span><br>")
        response_lines.append(f"• <strong>Sentimiento de Mercado (FinBERT)</strong>: <span style='color: {sent_color}; font-weight: bold;'>{sent_label}</span> (Score promedio: {sent_mean:+.2f})<br>")
        response_lines.append(f"• <strong>Volumen de Cobertura</strong>: {news_vol} artículos indexados.<br><br>")

        if citations:
            response_lines.append("<strong>Evidencia y Factores Clave Extraídos:</strong><br>")
            for c in citations:
                response_lines.append(f"• Según reportes de {c['source']} [{c['num']}], <em>\"{c['title']}\"</em>. El análisis de sentimiento clasifica este catalizador con confianza {c['confidence'].lower()}.<br>")
            response_lines.append("<br><em>Haz clic en las referencias numeradas para examinar el texto completo de las noticias citadas.</em>")
        else:
            response_lines.append("<em>No se encontraron artículos recientes para este activo en la base de datos de noticias.</em>")

    else:
        # Búsqueda temática / macroeconómica
        q_lower = query_text.lower()
        if any(term in q_lower for term in ['fed', 'tasa', 'rate', 'fomc', 'inflacion', 'inflation']):
            response_lines.append("<strong>Análisis Macroeconómico (Tasas de Interés y Reserva Federal):</strong><br><br>")
            response_lines.append("La política monetaria de la Reserva Federal impacta de manera directa el costo de capital y los múltiplos de valuación en el S&P 500. Nuestro modelo cuantitativo LightGBM captura estas condiciones a través de las 158 variables técnicas y de volatilidad (Alpha158).<br><br>")
        elif any(term in q_lower for term in ['mercado', 'market', 'sp500', 's&p', 'portafolio', 'portfolio']):
            response_lines.append("<strong>Visión General del Mercado (S&P 500 & Alpha158):</strong><br><br>")
            response_lines.append("El sistema FinancialRAG evalúa el universo de activos del S&P 500 integrando señales técnicas de series de tiempo financieras con análisis de sentimiento FinBERT.<br><br>")
        else:
            response_lines.append(f"<strong>Resultado de Recuperación para:</strong> \"{query_text}\"<br><br>")
            response_lines.append("He explorado el corpus de noticias financieras e información de mercado para tu consulta.<br><br>")

        if citations:
            response_lines.append("<strong>Noticias y Fuentes Relevantes Encontradas:</strong><br>")
            for c in citations:
                response_lines.append(f"• [{c['num']}] <strong>{c['title']}</strong> ({c['source']} · {c['date']})<br>")
            response_lines.append("<br><em>Puedes preguntar por un ticker específico (ej. AAPL, MSFT, TSLA, NVDA) para un desglose predictivo completo con IA.</em>")
        else:
            response_lines.append("No se localizaron artículos coincidentes en la base de datos. Prueba ingresando un ticker de acciones como <strong>AAPL</strong>, <strong>AMZN</strong> o <strong>NVDA</strong>.")

    return "".join(response_lines), citations


def execute_rag_query(query_text, session_key='', user_label='Analyst'):
    """
    Punto de entrada principal para procesar una consulta RAG:
    Ejecuta el pipeline de recuperación y síntesis, mide latencia y guarda en ChatQuery.
    """
    start_time = time.time()

    # 1. Identificar tickers
    tickers = extract_tickers_from_query(query_text)
    primary_ticker = tickers[0] if tickers else None

    # 2. Recuperar contexto
    context = retrieve_context(ticker=primary_ticker, query_text=query_text)

    # 3. Sintetizar respuesta con citas
    response_text, citations = synthesize_rag_response(query_text, context)

    latency_ms = int((time.time() - start_time) * 1000)

    # 4. Registrar en base de datos para Analytics
    try:
        ChatQuery.objects.create(
            query_text=query_text.strip(),
            session_key=session_key[:100] if session_key else '',
            response_text=response_text,
            sources_count=len(citations),
            latency_ms=max(latency_ms, 15),
            user_label=user_label
        )
    except Exception as e:
        print(f"Error registrando ChatQuery: {e}")

    return {
        'text': response_text,
        'citations': citations,
        'ticker': primary_ticker,
        'latency_ms': latency_ms
    }
