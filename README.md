# FinancialRAG - AI-Powered Financial News Assistant

Plataforma de predicción de mercado y asistente de noticias financieras impulsada por IA, construida con Django y arquitectura RAG (Retrieval-Augmented Generation).

## 🚀 Características Principales

### 🏠 Home - Asistente RAG
- Chat interactivo con análisis contextual de noticias
- Detección automática de tickers y empresas
- Respuestas basadas en predicciones ML reales
- Citas referenciadas a fuentes de noticias
- Análisis de sentimiento en tiempo real

### 📊 Portfolio - Gestión de Inversiones
- Predicciones del modelo ML (Alpha158 + LightGBM)
- Features de sentimiento (FinBERT)
- Gráfico de rendimiento con datos reales
- Retorno YTD calculado dinámicamente
- Correlación predicción vs actual

### 📰 News Feed - Noticias Financieras
- Artículos enriquecidos con sentimiento
- Filtros por categoría (Tech, Energy, Finance, Macro)
- Resúmenes AI generados automáticamente
- Datos de fuentes verificadas (Bloomberg, Reuters, WSJ)

### 📈 Analytics - Métricas del Sistema
- Rendimiento del modelo ML (IC, Rank IC, Sharpe)
- Volumen de predicciones por fecha
- Distribución de fuentes de noticias
- Métricas de uso del chat RAG

### ⚙️ Settings - Configuración
- Gestión de fuentes de noticias
- Preferencias de notificaciones
- Configuración del pipeline RAG

## 🏗️ Arquitectura

### Patrón MVT (Model-View-Template)
- **Model** (`models.py`): Modelos Django ORM
- **View** (`views.py`): Controladores HTTP + API endpoints
- **Template** (`templates/`): Interfaces HTML con Tailwind CSS

### Servicios ML
```
core/services/
├── ml_service.py           # Gestión de predicciones
├── sentiment_service.py    # Análisis de sentimiento
├── rag_service.py          # Motor RAG para chat
├── sentiment_features.py   # Extracción de features FinBERT
├── base_ml.py              # Funciones base LightGBM
── extended_ml_comparison.py # Comparación de variantes
```

## 📁 Estructura del Proyecto

```
financialrag/
├── core/                           # Aplicación Django principal
│   ├── models.py                   # Modelos de datos
│   │   ├── StockPrediction         # Predicciones ML
│   │   ├── SentimentFeature        # Features de sentimiento
│   │   ├── NewsArticle             # Artículos de noticias
│   │   └── ChatQuery               # Registro de consultas RAG
│   │
│   ├── views.py                    # Controladores HTTP
│   ├── urls.py                     # Rutas URL + API endpoints
│   ├── admin.py                    # Panel de administración
│   ├── tests.py                    # Tests unitarios
│   ├── context_processors.py       # Contexto global
│   │
│   ├── services/                   # Lógica de negocio ML
│   │   ├── ml_service.py           # Predicciones y métricas
│   │   ├── sentiment_service.py    # Sentimiento de noticias
│   │   ├── rag_service.py          # Motor RAG
│   │   ├── sentiment_features.py   # Extracción FinBERT
│   │   ├── base_ml.py              # Modelo LightGBM base
│   │   └── extended_ml_comparison.py # Comparación de modelos
│   │
│   ├── management/commands/        # Comandos de gestión
│   │   ├── load_predictions.py     # Cargar predicciones CSV
│   │   ├── load_sentiment.py       # Cargar noticias y sentimiento
│   │   ├── retrain_model.py        # Reentrenar modelo ML
│   │   └── enrich_financial_data.py # Enriquecer datos financieros
│   │
│   ├── templates/core/             # Templates HTML
│   │   ├── home.html               # Chat RAG
│   │   ├── portfolio.html          # Portafolio
│   │   ├── newsfeed.html           # Feed de noticias
│   │   ├── analytics.html          # Analíticas
│   │   └── settings.html           # Configuración
│   │
│   ├── static/                     # Assets estáticos
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
├── p1/                             # Configuración Django
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── manage.py                       # Utilidad Django
├── requirements.txt                # Dependencias Python
└── README.md                       # Documentación
```

## 🛠️ Tecnologías

### Backend
- **Django 6.1.1** - Framework web
- **Python 3.14** - Lenguaje de programación

### Machine Learning
- **LightGBM 4.7.0** - Modelo de predicción
- **PyTorch 2.14.0** - Deep learning
- **Transformers** - FinBERT para sentimiento
- **Scikit-learn** - Métricas y utilidades
- **NumPy & Pandas** - Procesamiento de datos

### Frontend
- **Tailwind CSS** - Framework CSS
- **Chart.js** - Visualización de datos
- **Inter & JetBrains Mono** - Tipografías

### Datos Financieros
- **yfinance** - Datos de mercado
- **TA-Lib** - Indicadores técnicos
- **Alpha158** - Features técnicas (158 factores)

## 🚦 Instalación y Ejecución

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

### 7. Acceder a la aplicación
Abrir http://127.0.0.1:8000/ en el navegador

## 📡 API Endpoints

### Predicciones
- `GET /api/predictions/` - Últimas predicciones
- `GET /api/predictions/<ticker>/` - Predicciones por ticker

### Sentimiento
- `GET /api/sentiment/` - Últimas features de sentimiento
- `GET /api/sentiment/<ticker>/` - Sentimiento por ticker

### Chat RAG
- `POST /api/chat/` - Enviar consulta al asistente RAG

### Métricas
- `GET /api/metrics/` - Métricas del modelo ML
- `GET /api/analytics/volume/` - Volumen de predicciones por fecha
- `GET /api/ticker/<ticker>/history/` - Historial de ticker

### Configuración
- `POST /api/settings/save/` - Guardar preferencias

## 🤖 Modelo ML

### Arquitectura
- **Base**: Alpha158 (158 features técnicas)
- **Modelo**: LightGBM Regressor
- **Entrenamiento**: 2008-2014 (train), 2015-2016 (valid), 2017-2026 (test)
- **Target**: `Ref($close, -2)/Ref($close, -1) - 1`

### Variantes del Modelo
1. **Variante A**: Alpha158 only (baseline)
2. **Variante B**: Alpha158 + sentiment_mean + news_volume
3. **Variante C**: Alpha158 + todas las features de sentimiento

### Métricas de Evaluación
- **IC (Information Coefficient)**: Correlación cross-sectional
- **Rank IC**: Correlación de rangos (Spearman)
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Pérdida máxima desde pico

### Features de Sentimiento
- `sentiment_mean`: Promedio de sentimiento
- `sentiment_max`: Máximo sentimiento
- `sentiment_min`: Mínimo sentimiento
- `sentiment_dispersion`: Desviación estándar
- `news_volume`: Número de noticias
- `positive_ratio`: Proporción positiva
- `negative_ratio`: Proporción negativa

## 🧪 Testing

```bash
# Ejecutar tests unitarios
python manage.py test core

# Tests específicos
python manage.py test core.tests.MLServiceTest
python manage.py test core.tests.RAGServiceTest
```

## 📊 Base de Datos

### Modelos

#### StockPrediction
- Predicciones del modelo ML por ticker y fecha
- Campos: ticker, date, predicted_return, actual_return, model_version

#### SentimentFeature
- Features de sentimiento agregadas
- Campos: ticker, date, sentiment_mean, sentiment_max, sentiment_dispersion, news_volume

#### NewsArticle
- Artículos de noticias financieras
- Campos: ticker, date, headline, text, sentiment_score, source, category

#### ChatQuery
- Registro de consultas del chat RAG
- Campos: query_text, session_key, response_text, sources_count, latency_ms

## 🔧 Comandos de Gestión

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

## 📝 Notas de Desarrollo

### Ramas
- `development` - Rama principal de desarrollo
- `jeremiasFigueroaGarcia` - Proyecto MovieReviews (separado)

### Repositorios
- **FinancialRAG**: Este proyecto (rama development)
- **MovieReviews**: Proyecto separado en rama jeremiasFigueroaGarcia

### Datos
- Las predicciones base están en `core/data/predictions.csv` (1,054,461 registros)
- Las noticias sintéticas se generan con `--generate-synthetic`
- El modelo FinBERT se descarga automáticamente la primera vez

## 📄 Licencia

Proyecto académico - EAFIT University

##  Autores

- Jeremías Figueroa García
- Desarrollo como proyecto final de curso

## 🙏 Agradecimientos

- Qlib - Framework de investigación cuantitativa
- FinBERT - Modelo de sentimiento financiero
- LightGBM - Gradient boosting framework
