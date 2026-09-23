from django.contrib import admin
from .models import StockPrediction, SentimentFeature, NewsArticle, ChatQuery


@admin.register(StockPrediction)
class StockPredictionAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'date', 'predicted_return', 'actual_return', 'model_version')
    list_filter = ('model_version', 'date')
    search_fields = ('ticker',)
    ordering = ('-date', 'ticker')


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'headline', 'source', 'date', 'sentiment_score', 'category')
    list_filter = ('source', 'category', 'date')
    search_fields = ('ticker', 'headline', 'text')
    ordering = ('-date', 'ticker')


@admin.register(SentimentFeature)
class SentimentFeatureAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'date', 'sentiment_mean', 'sentiment_max', 'sentiment_dispersion', 'news_volume')
    list_filter = ('date',)
    search_fields = ('ticker',)
    ordering = ('-date', 'ticker')


@admin.register(ChatQuery)
class ChatQueryAdmin(admin.ModelAdmin):
    list_display = ('query_text', 'user_label', 'latency_ms', 'sources_count', 'created_at')
    list_filter = ('user_label', 'created_at')
    search_fields = ('query_text', 'response_text')
    ordering = ('-created_at',)
