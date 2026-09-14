from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('portfolio/', views.portfolio, name='portfolio'),
    path('newsfeed/', views.newsfeed, name='newsfeed'),
    path('analytics/', views.analytics, name='analytics'),
    path('settings/', views.settings, name='settings'),
]
