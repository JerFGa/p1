import json
from datetime import date
from django.test import TestCase, Client
from core.models import StockPrediction, SentimentFeature, NewsArticle, ChatQuery


class FinancialRAGTests(TestCase):
    def setUp(self):
        self.client = Client()

        # Crear datos de prueba
        StockPrediction.objects.create(
            ticker='AAPL',
            date=date(2026, 2, 18),
            predicted_return=0.015,
            actual_return=0.012,
            model_version='alpha158_baseline'
        )
        StockPrediction.objects.create(
            ticker='MSFT',
            date=date(2026, 2, 18),
            predicted_return=-0.005,
            actual_return=-0.003,
            model_version='alpha158_baseline'
        )

        SentimentFeature.objects.create(
            ticker='AAPL',
            date=date(2026, 2, 18),
            sentiment_mean=0.65,
            sentiment_max=0.9,
            sentiment_dispersion=0.1,
            news_volume=5
        )

        NewsArticle.objects.create(
            ticker='AAPL',
            date=date(2026, 2, 18),
            headline='Apple announces record services revenue growth',
            text='Apple reported strong fiscal quarter results driven by record services revenue and expanding cloud ecosystem.',
            source='Bloomberg',
            sentiment_score=0.75,
            category='Earnings'
        )

    def test_home_page_loads(self):
        """Verifica que la página de inicio cargue correctamente con su contexto y pipeline."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('stocks', response.context)
        self.assertIn('pipeline_sources', response.context)
        self.assertGreater(response.context['pipeline_sources'], 0)

    def test_portfolio_page_and_metrics(self):
        """Verifica que el portafolio cargue métricas reales sin errores de agregado en querysets."""
        response = self.client.get('/portfolio/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('total_positions', response.context)
        self.assertEqual(response.context['total_positions'], 2)
        self.assertIn('perf_chart_json', response.context)

    def test_newsfeed_has_summaries_and_sentiment(self):
        """Verifica que el feed de noticias incluya textos de resumen y etiquetas de sentimiento."""
        response = self.client.get('/newsfeed/')
        self.assertEqual(response.status_code, 200)
        articles = response.context['articles']
        self.assertGreater(len(articles), 0)
        first = articles[0]
        self.assertTrue(len(first['summary']) > 0)
        self.assertIn('sentiment_label', first)
        self.assertEqual(first['sentiment_label'], 'Bullish')

    def test_analytics_recent_queries_log(self):
        """Verifica que la vista de analíticas reciba las consultas de ChatQuery."""
        ChatQuery.objects.create(
            query_text='How is Apple performing?',
            session_key='test-session',
            response_text='Analysis response',
            sources_count=2,
            latency_ms=45,
            user_label='Analyst'
        )

        response = self.client.get('/analytics/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('queries', response.context)
        self.assertEqual(len(response.context['queries']), 1)
        self.assertEqual(response.context['queries'][0]['text'], 'How is Apple performing?')

    def test_analytics_volume_api(self):
        """Verifica que el endpoint de volumen por rango responda correctamente."""
        response = self.client.get('/api/analytics/volume/?range=7d')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('data', data)
        self.assertGreater(len(data['data']), 0)

    def test_api_chat_rag_execution(self):
        """Verifica que el motor RAG responda con citas reales y registre la consulta en la BD."""
        initial_queries_count = ChatQuery.objects.count()

        payload = {'message': 'Tell me about Apple and its latest revenue growth'}
        response = self.client.post(
            '/api/chat/',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'ok')
        self.assertIn('AAPL', data['text'])
        self.assertIn('citations', data)
        self.assertGreater(len(data['citations']), 0)
        self.assertEqual(data['citations'][0]['source'], 'Bloomberg')

        # Verificar que se persistió en la BD
        self.assertEqual(ChatQuery.objects.count(), initial_queries_count + 1)

    def test_api_save_settings_session(self):
        """Verifica que las preferencias del usuario se guarden en la sesión."""
        payload = {
            'notif': False,
            'autoSumm': True,
            'darkMode': True,
            'active_sources': ['Bloomberg', 'WSJ']
        }
        response = self.client.post(
            '/api/settings/save/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'ok')
        self.assertFalse(data['settings']['notif'])
        self.assertEqual(data['settings']['active_sources'], ['Bloomberg', 'WSJ'])

    def test_api_ticker_history(self):
        """Verifica que el endpoint de historial de ticker retorne datos para gráficas."""
        response = self.client.get('/api/ticker/AAPL/history/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['ticker'], 'AAPL')
        self.assertIn('labels', data)
        self.assertIn('predicted_returns', data)
        self.assertIn('actual_returns', data)

