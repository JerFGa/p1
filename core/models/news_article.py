from django.db import models


class NewsArticle(models.Model):
    """Artículos de noticias para análisis de sentimiento."""
    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    headline = models.CharField(max_length=500)
    text = models.TextField()
    sentiment_score = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=100, blank=True)
    category = models.CharField(max_length=50, blank=True, default='General')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'core'
        ordering = ['-date', 'ticker']
        indexes = [
            models.Index(fields=['ticker', 'date']),
        ]

    def __str__(self):
        return f"{self.ticker} {self.date}: {self.headline[:50]}"
