from django.db import models


class StockPrediction(models.Model):
    """Predicciones del modelo ML para cada ticker."""
    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    predicted_return = models.FloatField()
    actual_return = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=50, default='alpha158_baseline')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'core'
        unique_together = ['ticker', 'date', 'model_version']
        ordering = ['-date', 'ticker']

    def __str__(self):
        return f"{self.ticker} {self.date}: {self.predicted_return:.4f}"
