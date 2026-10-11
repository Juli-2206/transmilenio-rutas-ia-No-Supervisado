# Sistema Inteligente de Rutas — K-Means Clustering TransMilenio

**Materia:** Inteligencia Artificial  
**Universidad:** Corporación Universitaria Iberoamericana  
**Estudiante:** Sebastian Sanchez Puerto  
**Repositorio:** https://github.com/Juli-2206/transmilenio-rutas-ia-No-Supervisado  

---

## Descripción General

Sistema de **aprendizaje no supervisado** aplicado al Sistema de Transporte Masivo TransMilenio de Bogotá D.C. El modelo K-Means descubre de forma autónoma agrupaciones naturales en los viajes, **sin utilizar etiquetas predefinidas**.

**Referencia teórica:** Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*. McGraw-Hill España. Capítulo 16: Técnicas de agrupamiento.

---

## Estructura del Repositorio

```
transmilenio-rutas-ia-No-Supervisado/
│
├── generar_dataset_clustering.py      ← Genera el dataset sin etiquetas
├── modelo_clustering.py               ← Modelo K-Means completo
├── dataset_clustering_transmilenio.csv  ← Dataset (484 registros, 8 features)
├── dataset_con_clusters.csv           ← Dataset con clusters asignados
├── clustering_panel.png               ← Panel 6 visualizaciones
├── clustering_radar.png               ← Perfiles radar de clusters
├── descripcion_datos_clustering.docx  ← Documento de descripción de datos
├── pruebas_modelo_clustering.docx     ← Documento de pruebas
└── README.md
```

---

## Aprendizaje No Supervisado: K-Means Clustering

### ¿Qué diferencia tiene con el aprendizaje supervisado?

| Aspecto | Árbol de Decisión (supervisado) | K-Means (no supervisado) |
|---------|--------------------------------|--------------------------|
| Usa etiquetas | ✅ Sí (3 categorías predefinidas) | ❌ No |
| Grupos predefinidos | ✅ Por el experto humano | ❌ Los descubre el algoritmo |
| Métrica de evaluación | Accuracy 85.95 % | Silhouette Score 0.2502 |
| Grupos encontrados | 3 (Rápida / Media / Larga) | 2 (Viaje Corto / Viaje Largo) |

---

## Dataset de Clustering

- **484 registros** de viajes sintéticos representativos de TransMilenio
- **8 variables numéricas** — sin columna de categoría/etiqueta
- Generado con semilla aleatoria fija (`np.random.seed(42)`) para reproducibilidad

### Variables del dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `troncal_origen_cod` | int (0–3) | Troncal de origen codificada |
| `troncal_destino_cod` | int (0–3) | Troncal de destino codificada |
| `hora_periodo` | int (0–5) | Período horario del viaje |
| `dia_tipo` | int (0–2) | Tipo de día (hábil, sábado, festivo) |
| `num_estaciones` | int | Número de estaciones en la ruta |
| `num_transbordos` | int (0–1) | Transbordos necesarios |
| `distancia_km` | float | Distancia aproximada del recorrido |
| `tiempo_minutos` | float | Tiempo real de viaje en minutos |

### Períodos horarios

| Código | Período | Factor velocidad |
|--------|---------|-----------------|
| 0 | Madrugada | ×1.10 (menor tráfico) |
| 1 | Mañana pico | ×0.70 (mayor congestión) |
| 2 | Mañana | ×0.95 |
| 3 | Tarde | ×0.85 |
| 4 | Tarde pico | ×0.72 (mayor congestión) |
| 5 | Noche | ×1.05 |

---

## Ejecución

### Paso 1 — Generar el dataset sin etiquetas

```bash
python generar_dataset_clustering.py
```

Genera `dataset_clustering_transmilenio.csv` con 484 registros.

### Paso 2 — Ejecutar el modelo K-Means

```bash
python modelo_clustering.py
```

Genera:
- `dataset_con_clusters.csv` — dataset con columna `cluster` asignada
- `clustering_panel.png` — 6 visualizaciones del modelo
- `clustering_radar.png` — perfiles radar de los clusters

---

## Metodología del Modelo

### 1. Normalización (StandardScaler)
Las 8 variables se normalizan antes de aplicar K-Means para evitar que variables con rangos amplios (como `tiempo_minutos`: 5–101 min) dominen sobre variables de rango pequeño (como `num_transbordos`: 0–1).

### 2. Determinación del K óptimo

**Método del Codo (Elbow):** se calcula la inercia (WCSS) para K = 2 a 8. La curva se aplana notablemente en K = 2.

**Coeficiente de Silhouette:** se calcula para cada K. El valor máximo ocurre en K = 2 (silhouette = 0.2502).

Ambos métodos coinciden: **K óptimo = 2**.

### 3. Modelo Final

```python
KMeans(
    n_clusters=2,
    init='k-means++',   # inicialización inteligente
    n_init=20,          # 20 inicializaciones para robustez
    max_iter=500,       # máximo de iteraciones
    random_state=42
)
```

### 4. Proyección PCA
Los 8 features se reducen a 2 dimensiones con PCA para visualización.  
Varianza explicada: **55.3 %** (PC1: 35.8 %, PC2: 19.5 %).

---

## Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| K óptimo (Elbow + Silhouette) | **2** |
| Inercia — WCSS | 2 751.01 |
| Silhouette Score | 0.2502 |
| Varianza explicada PCA 2D | 55.3 % |
| Iteraciones hasta convergencia | 11 |

### Clusters Descubiertos

| Cluster | Nombre | Registros | % del total | Tiempo medio | Estaciones medias | Distancia media |
|---------|--------|-----------|-------------|-------------|-------------------|-----------------|
| Cluster 0 | **Viaje Largo** | 275 | 56.8 % | 55 min | 14 estaciones | 9.2 km |
| Cluster 1 | **Viaje Corto** | 209 | 43.2 % | 22 min | 6 estaciones | 3.8 km |

### Interpretación de los Clusters

**Cluster 0 — Viaje Largo:**  
Predominan rutas inter-troncales con transbordos. La distancia supera los 8 km y el tiempo medio es de 55 minutos. Los viajes en hora pico (códigos 1 y 4) tienen mayor presencia.

**Cluster 1 — Viaje Corto:**  
Rutas dentro de la misma troncal, sin transbordos. La distancia media es inferior a 4 km y el tiempo promedio es 22 minutos. Proporcionalmente mayor presencia de viajes en madrugada y noche.

---

## Visualizaciones Generadas

### Panel de Clustering (`clustering_panel.png`)
6 gráficos en una sola imagen:
1. **Método del Codo** — inercia vs K
2. **Silhouette por K** — coeficiente de silueta para K = 2 a 8
3. **Dispersión PCA 2D** — proyección de los clusters
4. **Distribución de tiempo** — histograma por cluster
5. **Estaciones y distancia** — boxplots comparativos
6. **Silhouette por muestra** — coherencia interna de cada cluster

### Radar de Clusters (`clustering_radar.png`)
Gráfico polar con el perfil promedio normalizado de cada cluster en los 8 features. Permite comparar visualmente la "forma" de cada grupo.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Juli-2206/transmilenio-rutas-ia-No-Supervisado.git
cd transmilenio-rutas-ia-No-Supervisado

# Instalar dependencias
pip install pandas numpy matplotlib scikit-learn

# Ejecutar
python generar_dataset_clustering.py
python modelo_clustering.py
```

**Requisitos:** Python 3.8 o superior.

---

## Red de TransMilenio Utilizada

El dataset usa 4 troncales con 35 estaciones:

| Troncal | Código | Estaciones | Velocidad base | Longitud |
|---------|--------|------------|---------------|---------|
| Caracas | 0 | 15 | 28 km/h | 24.0 km |
| Américas | 1 | 8 | 30 km/h | 13.5 km |
| Calle 80 | 2 | 8 | 32 km/h | 12.0 km |
| Suba | 3 | 4 | 35 km/h | 8.5 km |

---

## Documentos Entregables

- **`descripcion_datos_clustering.docx`** — Fuentes de datos, estructura del dataset, estadísticas descriptivas y comparación con el dataset supervisado (Activity 4). Formato APA 7.ª edición.

- **`pruebas_modelo_clustering.docx`** — Parámetros del modelo, métricas de evaluación, 5 casos de prueba detallados con predicción y verificación, análisis de resultados y conclusiones.

---

## Referencia

Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*. McGraw-Hill España. Capítulo 16: Técnicas de agrupamiento, pp. 389–415.

---

## Licencia

Proyecto académico — Corporación Universitaria Iberoamericana, 2026.  
Uso exclusivo para fines educativos.
