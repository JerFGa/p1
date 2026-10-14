from django.db import models


class ChatQuery(models.Model):
    """Registro de consultas y respuestas generadas en el asistente RAG."""
    query_text = models.CharField(max_length=500)
    session_key = models.CharField(max_length=100, blank=True)
    response_text = models.TextField(blank=True)
    sources_count = models.IntegerField(default=0)
    latency_ms = models.IntegerField(default=0)
    user_label = models.CharField(max_length=50, default='Analyst')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'core'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.created_at.strftime('%Y-%m-%d %H:%M')}: {self.query_text[:40]}"
