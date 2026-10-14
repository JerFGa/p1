from django.db import models


class SentimentFeature(models.Model):
    """Features de sentimiento calculadas por ticker y fecha."""
    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    sentiment_mean = models.FloatField(default=0.0)
    sentiment_max = models.FloatField(default=0.0)
    sentiment_dispersion = models.FloatField(default=0.0)
    news_volume = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'core'
        unique_together = ['ticker', 'date']
        ordering = ['-date', 'ticker']

    def __str__(self):
        return f"{self.ticker} {self.date}: mean={self.sentiment_mean:.3f}"
