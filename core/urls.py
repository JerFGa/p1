from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('portfolio/', views.portfolio, name='portfolio'),
    path('newsfeed/', views.newsfeed, name='newsfeed'),
    path('analytics/', views.analytics, name='analytics'),
    path('settings/', views.settings, name='settings'),
    
    # API endpoints
    path('api/predictions/', views.api_predictions, name='api_predictions'),
    path('api/predictions/<str:ticker>/', views.api_predictions, name='api_predictions_ticker'),
    path('api/sentiment/', views.api_sentiment, name='api_sentiment'),
    path('api/sentiment/<str:ticker>/', views.api_sentiment, name='api_sentiment_ticker'),
    path('api/metrics/', views.api_metrics, name='api_metrics'),
]
