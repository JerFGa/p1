# FinancialRAG ML Services

Capa de servicios para el pipeline de Machine Learning de FinancialRAG.

## Estructura

```
services/
├── ml_service.py              # Script original Qlib + LightGBM
├── sentiment_features.py      # Módulo de features de sentimiento
├── extended_ml_comparison.py  # Comparación de 3 variantes del modelo
└── __init__.py
```

## Módulos

### 1. ml_service.py (Base)

Script original que implementa:
- Qlib Alpha158 features
- LightGBM Regressor
- Diagnóstico cross-sectional (IC, Rank IC, Sharpe, Drawdown)
- Logging a MLflow

**Uso:**
```python
from core.services.ml_service import main as run_baseline
run_baseline()
```

### 2. sentiment_features.py

Extiende el pipeline con features de sentimiento de noticias:
- Carga dataset de noticias financieras (FNSPID o sintético)
- Infiere sentimiento con FinBERT (ProsusAI/finbert)
- Cachea resultados en parquet para reproducibilidad
- Calcula features por (instrument, fecha):
  - `sentiment_mean`: Promedio de sentimiento
  - `sentiment_max`: Máximo sentimiento
  - `sentiment_dispersion`: Desviación estándar
  - `news_volume`: Número de noticias

**Uso:**
```python
from core.services.sentiment_features import get_sentiment_features

sentiment_features = get_sentiment_features(
    instruments=["AAPL", "MSFT", "AMZN"],
    date_range=("2008-01-01", "2026-02-20")
)
```

### 3. extended_ml_comparison.py

Compara 3 variantes del modelo:
1. **Alpha158 Only** (baseline)
2. **Alpha158 + Core Sentiment** (sentiment_mean + news_volume)
3. **Alpha158 + Full Sentiment** (todas las features de sentimiento)

Genera:
- Tabla comparativa de métricas (IC, Sharpe, Return, Drawdown)
- Análisis SHAP para la variante con todas las features
- Reporte en Markdown (`comparison_report.md`)

**Uso:**
```python
from core.services.extended_ml_comparison import main as run_comparison
run_comparison()
```

## Datos

Los datos del modelo se almacenan en `core/data/`:

```
data/
├── predictions.csv              # Predicciones del modelo
├── feature_importance.csv       # Importancia de features
├── sentiment_cache/             # Cache de FinBERT
│   └── finbert_sentiment.parquet
└── financial_news.csv           # Dataset de noticias (opcional)
```

### Dataset de Noticias

Para usar noticias reales, coloca `financial_news.csv` en `core/data/` con columnas:
- `date`: Fecha de la noticia (YYYY-MM-DD)
- `ticker`: Ticker del instrumento (ej: AAPL)
- `headline`: Título de la noticia
- `text`: Texto completo de la noticia

Si no existe, se usa un dataset sintético para demostración.

## Dependencias

Agregadas a `requirements.txt`:
- `transformers==5.5.4` - Para FinBERT
- `torch==2.10.0` - Backend de transformers
- `datasets==5.0.1` - Carga de datasets
- `shap==0.46.0` - Análisis de importancia de features
- `pyarrow==21.0.0` - Lectura/escritura de parquet

## Ejecución

### Ejecutar baseline (Alpha158 only)
```bash
cd financialrag
source venv/bin/activate
python -m core.services.ml_service
```

### Ejecutar comparación completa
```bash
cd financialrag
source venv/bin/activate
python -m core.services.extended_ml_comparison
```

## Resultados

Los resultados se guardan en `core/services/model_output/`:
- `comparison_report.md` - Reporte comparativo
- `shap_values_*.csv` - Valores SHAP por variante
- `shap_importance_*.png` - Gráficos de importancia
- Métricas cross-sectional por variante

## Limitaciones

1. **Cobertura de noticias**: No todos los tickers tienen la misma cobertura
2. **Periodos de alta volatilidad**: El sentimiento puede ser lagging
3. **Datos sintéticos**: Para demostración, reemplazar con FNSPID real
4. **Timestamps**: La precisión afecta la alineación de features

## Próximos Pasos

1. Reemplazar dataset sintético con FNSPID real de Kaggle
2. Agregar más features de sentimiento (volatilidad, momentum de sentimiento)
3. Implementar backtesting completo con Qlib
4. Integrar predicciones en la UI de Django
