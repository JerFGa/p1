from django.shortcuts import render

def home(request):
    return render(request, 'core/home.html')

def portfolio(request):
    stocks = [
        {'ticker': 'AAPL', 'name': 'Apple Inc.', 'shares': 12, 'price': '196.34', 'change': 2.14, 'sentiment': 'Bullish', 'sentiment_bg': '#071f0e', 'sentiment_color': '#6ee7b7', 'sentiment_border': '#064e3b'},
        {'ticker': 'MSFT', 'name': 'Microsoft', 'shares': 8, 'price': '418.72', 'change': 0.87, 'sentiment': 'Bullish', 'sentiment_bg': '#071f0e', 'sentiment_color': '#6ee7b7', 'sentiment_border': '#064e3b'},
        {'ticker': 'AMZN', 'name': 'Amazon', 'shares': 15, 'price': '228.45', 'change': 3.51, 'sentiment': 'Strong Buy', 'sentiment_bg': '#052e16', 'sentiment_color': '#34d399', 'sentiment_border': '#059669'},
        {'ticker': 'TSLA', 'name': 'Tesla Inc.', 'shares': 20, 'price': '234.80', 'change': -1.22, 'sentiment': 'Neutral', 'sentiment_bg': '#1c1917', 'sentiment_color': '#a8a29e', 'sentiment_border': '#44403c'},
        {'ticker': 'NVDA', 'name': 'NVIDIA Corp.', 'shares': 5, 'price': '875.40', 'change': 4.30, 'sentiment': 'Bullish', 'sentiment_bg': '#071f0e', 'sentiment_color': '#6ee7b7', 'sentiment_border': '#064e3b'},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'shares': 10, 'price': '176.20', 'change': 0.44, 'sentiment': 'Bullish', 'sentiment_bg': '#071f0e', 'sentiment_color': '#6ee7b7', 'sentiment_border': '#064e3b'},
    ]
    return render(request, 'core/portfolio.html', {'stocks': stocks})

def newsfeed(request):
    articles = [
        {'id': 1, 'category': 'Tech', 'ticker': 'AMZN', 'headline': 'AWS Revenue Hits Record $24.2B in Q4 on AI Infrastructure Surge', 'source': 'Reuters', 'time': '2h ago', 'summary': 'AWS segment grew 19% YoY. Primary driver: Bedrock AI platform uptake. Risk: potential margin compression from infrastructure capex. Sentiment: Bullish.'},
        {'id': 2, 'category': 'Finance', 'ticker': 'FED', 'headline': 'Fed Holds Rates Steady, Powell Signals Patient Approach to Cuts', 'source': 'Bloomberg', 'time': '3h ago', 'summary': 'No imminent rate cuts. Bond markets repriced with 10Y yields rising 8bps. Dollar index strengthened. Macro sentiment: Cautious.'},
        {'id': 3, 'category': 'Tech', 'ticker': 'NVDA', 'headline': 'NVIDIA Blackwell GPU Demand Exceeds Supply Through Mid-2025', 'source': 'WSJ', 'time': '4h ago', 'summary': 'Supply bottleneck is bullish for NVDA pricing power but could delay customer AI deployments. NVDA consensus target raised to $950. Sentiment: Strong Buy.'},
        {'id': 4, 'category': 'Energy', 'ticker': 'XOM', 'headline': 'ExxonMobil Cuts Capex Forecast Amid Oil Price Volatility', 'source': 'FT', 'time': '5h ago', 'summary': 'XOM capex cut signals defensive posture. Oil bear case gaining traction below $75/bbl. Energy sector sentiment: Neutral.'},
        {'id': 5, 'category': 'Tech', 'ticker': 'AAPL', 'headline': 'Apple Vision Pro Developer Adoption Doubles in Q4 2024', 'source': 'TechCrunch', 'time': '6h ago', 'summary': 'Positive signal for Apple services ecosystem long-term. Vision Pro TAM remains uncertain. Sentiment: Mildly Bullish.'},
        {'id': 6, 'category': 'Macro', 'ticker': 'USD', 'headline': 'US Jobless Claims Fall to 201K, Lowest Since January 2023', 'source': 'CNBC', 'time': '7h ago', 'summary': 'Strong labor market limits Fed flexibility on rate cuts. Equities mixed reaction. Macro risk: higher-for-longer rates.'},
        {'id': 7, 'category': 'Finance', 'ticker': 'JPM', 'headline': 'JPMorgan Q4 Net Income Rises 50% on Higher Rates, Trading Revenue', 'source': 'Bloomberg', 'time': '9h ago', 'summary': 'JPM results reinforce bank sector recovery thesis. Credit quality remains benign. Finance sentiment: Bullish.'},
    ]
    return render(request, 'core/newsfeed.html', {'articles': articles})

def analytics(request):
    queries = [
        {'text': 'Analyze AMZN', 'time': '09:41', 'latency': '1.2s', 'sources': 3, 'user': 'Session #4821'},
        {'text': 'Fed Meeting Summary', 'time': '09:38', 'latency': '0.9s', 'sources': 2, 'user': 'Session #4821'},
        {'text': 'TSLA earnings outlook', 'time': '09:22', 'latency': '1.6s', 'sources': 2, 'user': 'Session #4820'},
        {'text': 'Compare AAPL vs MSFT news', 'time': '09:05', 'latency': '2.1s', 'sources': 5, 'user': 'Session #4819'},
    ]
    return render(request, 'core/analytics.html', {'queries': queries})

def settings(request):
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
