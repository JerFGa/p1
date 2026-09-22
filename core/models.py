from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class StockPrediction(models.Model):
    """Predicciones del modelo ML para cada ticker."""
    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    predicted_return = models.FloatField()
    actual_return = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=50, default='alpha158_baseline')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['ticker', 'date', 'model_version']
        ordering = ['-date', 'ticker']

    def __str__(self):
        return f"{self.ticker} {self.date}: {self.predicted_return:.4f}"


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
        unique_together = ['ticker', 'date']
        ordering = ['-date', 'ticker']

    def __str__(self):
        return f"{self.ticker} {self.date}: mean={self.sentiment_mean:.3f}"


class NewsArticle(models.Model):
    """Artículos de noticias para análisis de sentimiento."""
    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    headline = models.CharField(max_length=500)
    text = models.TextField()
    sentiment_score = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date', 'ticker']
        indexes = [
            models.Index(fields=['ticker', 'date']),
        ]

    def __str__(self):
        return f"{self.ticker} {self.date}: {self.headline[:50]}"
