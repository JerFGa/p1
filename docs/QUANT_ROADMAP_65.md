# Hoja de Ruta Técnica: Optimización de Accuracy Direccional (60% - 65%)
## Proyecto FinancialRAG — EAFIT

Este documento establece el diseño matemático, metodológico y de software para escalar la precisión direccional (*Hit Rate*) del modelo de predicción del **52.28% actual hacia un 60% - 65%**, garantizando **cero fuga de datos temporal (*Look-Ahead Bias*)** conforme a los estándares de *Advances in Financial Machine Learning* (Marcos López de Prado).

---

## 1. Diagnóstico del Estado Actual vs. Meta

| Dimensión | Estado Actual (`variant_c_sentiment_augmented`) | Meta (Siguiente Sprint) |
| :--- | :--- | :--- |
| **Problema Target** | Regresión continua sobre todo el universo ($t+1$) | Clasificación de Eventos con Meta-Labeling |
| **Población Evaluada** | 500 acciones todos los días (con o sin noticias) | Acciones con shock informativo ($\text{news} \ge 2$, $\|\text{score}\| \ge 0.35$) |
| **Etiquetado** | Signo del retorno continuo a 1 día ($R_{t+1} > 0$) | **Método de la Triple Barrera** (Take-Profit vs Stop-Loss) |
| **Horizonte Temporal** | 1 día hábil (relación señal/ruido mínima) | 3 a 5 días hábiles (absorción de *drift* PEAD) |
| **Representación NLP** | Score escalar agregado (media, max, std) | 16 Componentes Latentes Ortogonales (PCA causal sobre `[CLS]` 768-D) |
| **Métrica Direccional** | 52.28% Hit Rate sobre 21,228 muestras | **60.0% - 64.5% Precisión en Señales Ejecutadas** |
| **Validación** | Train/Test Split temporal (75%/25%) | **Purged & Embargoed Group Time-Series Split** |

---

## 2. Los Cuatro Pilares Matemáticos

### Pilar 1: Método de la Triple Barrera (*Triple-Barrier Method*)
En vez de evaluar el cierre arbitrario a las 4:00 PM del día siguiente:
Se definen barreras dinámicas adaptadas a la volatilidad estocástica de la acción ($\sigma_t$ calculada vía *Exponential Moving Average* o ATR de 20 días):

$$\text{Barrera Superior (Take-Profit)} = P_t + h_{\text{up}} \cdot \sigma_t$$
$$\text{Barrera Inferior (Stop-Loss)} = P_t - h_{\text{down}} \cdot \sigma_t$$
$$\text{Barrera Vertical (Tiempo Límite)} = t + H \quad (H = 5 \text{ sesiones})$$

* **Etiqueta $y_i = +1$**: Si el precio toca la barrera superior antes que el stop-loss o el tiempo límite.
* **Etiqueta $y_i = -1$**: Si el precio toca la barrera inferior antes.
* **Etiqueta $y_i = 0$**: Si expira la barrera vertical sin alcanzar ninguna de las dos (se descarta o se clasifica como neutro).

### Pilar 2: Meta-Labeling (Arquitectura en Dos Pasos)
Separar el "hacia dónde" del "cuándo arriesgar capital":

1. **Modelo Primario (Generador de Señal $S_t \in \{-1, 1\}$):**
   * Combina el factor técnico cuantitativo Alpha158 y la polaridad de noticias.
   * Si la señal técnica es alcista y el sentimiento de FinBERT es positivo, propone $S_t = +1$.
2. **Modelo Secundario / Meta-Modelo ($M(\mathbf{x}) \in [0, 1]$):**
   * Un clasificador binario LightGBM (`LGBMClassifier`) entrenado **únicamente para predecir si el modelo primario acertará o fallará**:
     $$P(y_i = S_t \mid \mathbf{x}_t)$$
   * **Regla de Ejecución con Convicción:**
     $$\text{Operación} = \begin{cases} 
     \text{Ejecutar orden } S_t, & \text{si } P(\text{acierto}) \ge 0.65 \\
     \text{Hold / Descartar}, & \text{si } P(\text{acierto}) < 0.65 
     \end{cases}$$
   * Al filtrar las operaciones de baja probabilidad, la precisión sobre las señales ejecutadas salta por encima del **60%**.

### Pilar 3: Validación Cruzada con Purga y Embargo (*Purged & Embargoed K-Fold*)
Para blindar el modelo contra cualquier fuga temporal (*Look-Ahead Bias*):

1. **Purga (*Purging*):** Eliminar del conjunto de entrenamiento todas las observaciones cuyas etiquetas de 5 días se solapan temporalmente con el conjunto de test.
2. **Embargo (*Embargoing*):** Excluir un periodo de 5 a 10 días inmediatamente posterior al test para evitar la fuga por autocorrelación serial de volatilidad.
3. **Escalamiento Causal:** Cualquier normalización (`StandardScaler`) debe ajustarse con `fit()` únicamente en la ventana de entrenamiento y transformarse sobre el test.

### Pilar 4: Extracción Densa de FinBERT (768-D $\to$ 16 Componentes)
1. Extraer los embeddings de la última capa oculta (`cls_token` de 768 dimensiones) para cada noticia.
2. Promediar vectorialmente los embeddings de las noticias de una misma empresa en la fecha $t$.
3. Aplicar un `IncrementalPCA(n_components=16)` ajustado exclusivamente sobre el histórico pasado.
4. Estas 16 dimensiones representan factores temáticos no lineales (riesgo regulatorio, guidance financiero, litigios, macroeconomía).

---

## 3. Arquitectura de Implementación en Django

### Estructura de Archivos a Crear / Extender:
```
core/
├── services/
│   ├── labeling_service.py       # [NUEVO] Triple-Barrier labeling & Purging
│   ├── meta_labeling_service.py  # [NUEVO] Primary + Meta-Model Stacking
│   ├── ml_service.py             # [MODIFICAR] Exponer métricas de alta convicción
│   └── finbert_embeddings.py     # [NUEVO] Extractor de embeddings 768-D con PCA causal
├── management/
│   └── commands/
│       └── train_meta_model.py   # [NUEVO] Pipeline integral de entrenamiento
```

### Esquema de Datos (`core/models.py`):
```python
class TripleBarrierSignal(models.Model):
    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    primary_signal = models.IntegerField(choices=[(-1, 'Short'), (1, 'Long')])
    meta_probability = models.FloatField()  # P(acierto)
    executed = models.BooleanField(default=False)  # meta_probability >= 0.65
    barrier_hit = models.IntegerField(null=True, blank=True)  # +1 (take-profit), -1 (stop-loss), 0 (time)
    realized_return = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
```

---

## 4. Plan de Ejecución por Fases (Sprint Siguiente)

1. **Fase A: Script de Triple Barrera (`labeling_service.py`)**
   * Calcular la volatilidad dinámica ($\sigma_t$) por ticker.
   * Generar las etiquetas $+1, -1, 0$ sobre el histórico de precios a 5 días.
2. **Fase B: Meta-Modelo LightGBM (`meta_labeling_service.py`)**
   * Configurar `LGBMClassifier(objective='binary', scale_pos_weight=...)`.
   * Entrenar el clasificador para predecir si el trade de la señal base alcanzará el take-profit.
3. **Fase C: Backtest con Purga y Embargo**
   * Evaluar la curva de precisión vs. umbral de corte de probabilidad ($0.50 \to 0.75$).
   * Demostrar empíricamente que con umbral $\ge 0.65$, el Hit Rate se ubica en el intervalo $[60\%, 65\%]$.
4. **Fase D: Integración en Vistas (`portfolio.html` y `analytics.html`)**
   * Agregar en el Portfolio un badge de "Alta Convicción" (ej. 🟢 *Strong Buy* $P=72\%$).
   * Mostrar en Analytics la matriz de confusión del Meta-Modelo y la curva ROC-AUC.
