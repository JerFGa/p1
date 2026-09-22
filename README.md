# FinancialRAG - AI-Powered Financial News Assistant

Plataforma de predicción de mercado y asistente de noticias financieras impulsada por IA, construida con Django y arquitectura RAG (Retrieval-Augmented Generation).

## Caracteristicas Principales

### Home - Asistente RAG
- Chat interactivo con analisis contextual de noticias
- Deteccion automatica de tickers y empresas
- Respuestas basadas en predicciones ML reales
- Citas referenciadas a fuentes de noticias
- Analisis de sentimiento en tiempo real

### Portfolio - Gestion de Inversiones
- Predicciones del modelo ML (Alpha158 + LightGBM)
- Features de sentimiento (FinBERT)
- Grafico de rendimiento con datos reales
- Retorno YTD calculado dinamicamente
- Correlacion prediccion vs actual

### News Feed - Noticias Financieras
- Articulos enriquecidos con sentimiento
- Filtros por categoria (Tech, Energy, Finance, Macro)
- Resumenes AI generados automaticamente
- Datos de fuentes verificadas (Bloomberg, Reuters, WSJ)

### Analytics - Metricas del Sistema
- Rendimiento del modelo ML (IC, Rank IC, Sharpe)
- Volumen de predicciones por fecha
- Distribucion de fuentes de noticias
- Metricas de uso del chat RAG

### Settings - Configuracion
- Gestion de fuentes de noticias
- Preferencias de notificaciones
- Configuracion del pipeline RAG

## Arquitectura

### Patron MVT (Model-View-Template)
- **Model** (`models.py`): Modelos Django ORM
- **View** (`views.py`): Controladores HTTP + API endpoints
- **Template** (`templates/`): Interfaces HTML con Tailwind CSS

### Servicios ML
```
core/services/
├── ml_service.py           # Gestion de predicciones
├── sentiment_service.py    # Analisis de sentimiento
├── rag_service.py          # Motor RAG para chat
├── sentiment_features.py   # Extraccion de features FinBERT
├── base_ml.py              # Funciones base LightGBM
└── extended_ml_comparison.py # Comparacion de variantes
```

## Estructura del Proyecto

```
financialrag/
├── core/                           # Aplicacion Django principal
│   ├── models.py                   # Modelos de datos
│   │   ├── StockPrediction         # Predicciones ML
│   │   ├── SentimentFeature        # Features de sentimiento
│   │   ├── NewsArticle             # Articulos de noticias
│   │   └── ChatQuery               # Registro de consultas RAG
│   │
│   ├── views.py                    # Controladores HTTP
│   ├── urls.py                     # Rutas URL + API endpoints
│   ├── admin.py                    # Panel de administracion
│   ├── tests.py                    # Tests unitarios
│   ├── context_processors.py       # Contexto global
│   │
│   ├── services/                   # Logica de negocio ML
│   │   ├── ml_service.py           # Predicciones y metricas
│   │   ├── sentiment_service.py    # Sentimiento de noticias
│   │   ├── rag_service.py          # Motor RAG
│   │   ├── sentiment_features.py   # Extraccion FinBERT
│   │   ├── base_ml.py              # Modelo LightGBM base
│   │   └── extended_ml_comparison.py # Comparacion de modelos
│   │
│   ├── management/commands/        # Comandos de gestion
│   │   ├── load_predictions.py     # Cargar predicciones CSV
│   │   ├── load_sentiment.py       # Cargar noticias y sentimiento
│   │   ├── retrain_model.py        # Reentrenar modelo ML
│   │   └── enrich_financial_data.py # Enriquecer datos financieros
│   │
│   ├── templates/core/             # Templates HTML
│   │   ├── home.html               # Chat RAG
│   │   ├── portfolio.html          # Portafolio
│   │   ├── newsfeed.html           # Feed de noticias
│   │   ├── analytics.html          # Analiticas
│   │   └── settings.html           # Configuracion
│   │
│   ├── static/                     # Assets estaticos
│   │   ├── css/style.css
│   │   └── js/app.js
│   │
│   ├── data/                       # Datos del modelo
│   │   ├── predictions.csv         # Predicciones base
│   │   ├── feature_importance.csv  # Importancia de features
│   │   └── sentiment_cache/        # Cache FinBERT
│   │
│   └── migrations/                 # Migraciones de BD
│
── p1/                             # Configuracion Django
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── manage.py                       # Utilidad Django
├── requirements.txt                # Dependencias Python
└── README.md                       # Documentacion
```

## Tecnologias

### Backend
- **Django 6.1.1** - Framework web
- **Python 3.14** - Lenguaje de programacion

### Machine Learning
- **LightGBM 4.7.0** - Modelo de prediccion
- **PyTorch 2.14.0** - Deep learning
- **Transformers** - FinBERT para sentimiento
- **Scikit-learn** - Metricas y utilidades
- **NumPy & Pandas** - Procesamiento de datos

### Frontend
- **Tailwind CSS** - Framework CSS
- **Chart.js** - Visualizacion de datos
- **Inter & JetBrains Mono** - Tipografias

### Datos Financieros
- **yfinance** - Datos de mercado
- **TA-Lib** - Indicadores tecnicos
- **Alpha158** - Features tecnicas (158 factores)

## Instalacion y Ejecucion

### 1. Clonar el repositorio
```bash
git clone https://github.com/JerFGa/p1.git
cd p1/financialrag
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar migraciones
```bash
python manage.py migrate
```

### 5. Cargar datos iniciales
```bash
# Cargar predicciones del modelo
python manage.py load_predictions --file core/data/predictions.csv

# Cargar noticias y generar sentimiento
python manage.py load_sentiment --generate-synthetic

# (Opcional) Enriquecer datos financieros
python manage.py enrich_financial_data
```

### 6. Iniciar servidor de desarrollo
```bash
python manage.py runserver
```

### 7. Acceder a la aplicacion
Abrir http://127.0.0.1:8000/ en el navegador

## API Endpoints

### Predicciones
- `GET /api/predictions/` - Ultimas predicciones
- `GET /api/predictions/<ticker>/` - Predicciones por ticker

### Sentimiento
- `GET /api/sentiment/` - Ultimas features de sentimiento
- `GET /api/sentiment/<ticker>/` - Sentimiento por ticker

### Chat RAG
- `POST /api/chat/` - Enviar consulta al asistente RAG

### Metricas
- `GET /api/metrics/` - Metricas del modelo ML
- `GET /api/analytics/volume/` - Volumen de predicciones por fecha
- `GET /api/ticker/<ticker>/history/` - Historial de ticker

### Configuracion
- `POST /api/settings/save/` - Guardar preferencias

## Modelo ML

### Arquitectura
- **Base**: Alpha158 (158 features tecnicas)
- **Modelo**: LightGBM Regressor
- **Entrenamiento**: 2008-2014 (train), 2015-2016 (valid), 2017-2026 (test)
- **Target**: `Ref($close, -2)/Ref($close, -1) - 1`

### Variantes del Modelo
1. **Variante A**: Alpha158 only (baseline)
2. **Variante B**: Alpha158 + sentiment_mean + news_volume
3. **Variante C**: Alpha158 + todas las features de sentimiento

### Metricas de Evaluacion
- **IC (Information Coefficient)**: Correlacion cross-sectional
- **Rank IC**: Correlacion de rangos (Spearman)
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Perdida maxima desde pico

### Features de Sentimiento
- `sentiment_mean`: Promedio de sentimiento
- `sentiment_max`: Maximo sentimiento
- `sentiment_min`: Minimo sentimiento
- `sentiment_dispersion`: Desviacion estandar
- `news_volume`: Numero de noticias
- `positive_ratio`: Proporcion positiva
- `negative_ratio`: Proporcion negativa

## Testing

```bash
# Ejecutar tests unitarios
python manage.py test core

# Tests especificos
python manage.py test core.tests.MLServiceTest
python manage.py test core.tests.RAGServiceTest
```

## Base de Datos

### Modelos

#### StockPrediction
- Predicciones del modelo ML por ticker y fecha
- Campos: ticker, date, predicted_return, actual_return, model_version

#### SentimentFeature
- Features de sentimiento agregadas
- Campos: ticker, date, sentiment_mean, sentiment_max, sentiment_dispersion, news_volume

#### NewsArticle
- Articulos de noticias financieras
- Campos: ticker, date, headline, text, sentiment_score, source, category

#### ChatQuery
- Registro de consultas del chat RAG
- Campos: query_text, session_key, response_text, sources_count, latency_ms

## Comandos de Gestion

```bash
# Cargar predicciones desde CSV
python manage.py load_predictions --file core/data/predictions.csv

# Cargar noticias y generar sentimiento
python manage.py load_sentiment --generate-synthetic

# Reentrenar modelo con sentiment features
python manage.py retrain_model --variant c --clear-existing

# Enriquecer datos financieros
python manage.py enrich_financial_data
```

## Licencia

Proyecto academico - EAFIT University

## Autores

<!-- Agregar autores aqui -->
