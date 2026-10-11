"""
=================================================================
SISTEMA INTELIGENTE DE RUTAS - TRANSMILENIO BOGOTÁ
Módulo de Aprendizaje No Supervisado: K-Means Clustering
=================================================================
Universidad : Corporación Universitaria Iberoamericana
Materia     : Inteligencia Artificial
Referencia  : Palma Méndez, J. T. (2008). Inteligencia artificial:
              métodos, técnicas y aplicaciones. McGraw-Hill España.
              Capítulo 16: Técnicas de agrupamiento.

Descripción:
  Aplica el algoritmo K-Means para descubrir agrupaciones
  naturales en los viajes de TransMilenio, SIN utilizar
  etiquetas previas. El modelo identifica patrones de viaje
  basándose únicamente en las características del trayecto.

Técnicas aplicadas:
  - Normalización (StandardScaler)
  - Método del Codo (Elbow) para determinar K óptimo
  - Análisis de Silhouette para validar cohesión de clusters
  - K-Means (Lloyd's algorithm)
  - Reducción PCA para visualización 2D
  - Análisis de perfiles de clusters (centroides)
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

print("=" * 65)
print("  MODELO K-MEANS CLUSTERING — TRANSMILENIO BOGOTÁ")
print("  Ref: Palma Méndez (2008), Cap. 16 — Técnicas de agrupamiento")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# 1. CARGA Y EXPLORACIÓN
# ══════════════════════════════════════════════════════════════════
print("\n[1] Cargando dataset...")
df = pd.read_csv("dataset_clustering_transmilenio.csv")

print(f"    Registros  : {len(df)}")
print(f"    Variables  : {df.shape[1]}")
print(f"\n    Vista previa:")
print(df[['troncal_origen','hora_periodo','num_estaciones',
          'num_transbordos','distancia_km','tiempo_minutos']].head(5).to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# 2. SELECCIÓN DE FEATURES Y NORMALIZACIÓN
# ══════════════════════════════════════════════════════════════════
print("\n[2] Preparando features para clustering...")

# Features numéricas que describen el viaje (SIN etiqueta de clase)
FEATURES = ['troncal_origen_cod', 'troncal_destino_cod',
            'hora_periodo', 'dia_tipo',
            'num_estaciones', 'num_transbordos',
            'distancia_km', 'tiempo_minutos']

NOMBRES_FEAT = ['Troncal Origen', 'Troncal Destino',
                'Hora Período', 'Día Tipo',
                'N° Estaciones', 'N° Transbordos',
                'Distancia (km)', 'Tiempo (min)']

X = df[FEATURES].values

# Normalización: escala todas las variables a media=0, std=1
# Necesario para K-Means — evita que variables de mayor magnitud
# (tiempo, distancia) dominen la medida de distancia euclidiana
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"    Features seleccionadas : {len(FEATURES)}")
print(f"    Normalización          : StandardScaler (media=0, σ=1)")
print(f"    Forma de X escalada    : {X_scaled.shape}")

# ══════════════════════════════════════════════════════════════════
# 3. MÉTODO DEL CODO — DETERMINAR K ÓPTIMO
# ══════════════════════════════════════════════════════════════════
print("\n[3] Método del Codo (Elbow) — buscando K óptimo...")

inercias    = []
silhouettes = []
K_RANGO     = range(2, 9)

for k in K_RANGO:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                max_iter=300, random_state=42)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(sil)
    print(f"    K={k}  Inercia={km.inertia_:8.1f}  Silhouette={sil:.4f}")

# K óptimo: mayor silhouette
k_optimo = K_RANGO.start + np.argmax(silhouettes)
print(f"\n    → K óptimo seleccionado: {k_optimo}  "
      f"(Silhouette={max(silhouettes):.4f})")

# ══════════════════════════════════════════════════════════════════
# 4. MODELO K-MEANS FINAL
# ══════════════════════════════════════════════════════════════════
print(f"\n[4] Entrenando K-Means con K={k_optimo}...")

modelo = KMeans(n_clusters=k_optimo, init='k-means++', n_init=20,
                max_iter=500, random_state=42)
modelo.fit(X_scaled)
labels    = modelo.labels_
inercia   = modelo.inertia_
sil_final = silhouette_score(X_scaled, labels)

print(f"    Inercia final      : {inercia:.2f}")
print(f"    Silhouette score   : {sil_final:.4f}")
print(f"    Iteraciones        : {modelo.n_iter_}")

# Distribución de clusters
print(f"\n    Distribución de clusters:")
for c in range(k_optimo):
    n = (labels == c).sum()
    print(f"      Cluster {c}: {n:3d} registros ({n/len(labels)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# 5. PERFILES DE CLUSTERS (centroides en escala original)
# ══════════════════════════════════════════════════════════════════
print(f"\n[5] Perfiles de clusters (centroides en escala original):")

centroides_orig = scaler.inverse_transform(modelo.cluster_centers_)
df_centros = pd.DataFrame(centroides_orig, columns=FEATURES)
df_centros.index = [f"Cluster {i}" for i in range(k_optimo)]

# Agregar etiqueta interpretada
HORA_LABELS = ["Madrugada","Mañana pico","Mañana","Tarde","Tarde pico","Noche"]
DIA_LABELS  = ["Hábil","Sábado","Festivo"]

print(f"\n  {'':12s} {'Estac':>6} {'Transbord':>10} {'Dist(km)':>9} {'Tiempo(min)':>12} {'Hora':>14} {'Día':>9}")
print("  " + "─" * 66)

nombres_cluster = []
for i, row in df_centros.iterrows():
    h_label = HORA_LABELS[int(round(row['hora_periodo']))]
    d_label = DIA_LABELS[int(round(row['dia_tipo']))]
    # Nombre interpretado del cluster
    t = row['tiempo_minutos']
    if t < 25:
        nombre = "Viaje Corto"
    elif t < 50:
        nombre = "Viaje Medio"
    else:
        nombre = "Viaje Largo"
    nombres_cluster.append(nombre)
    print(f"  {i} ({nombre:10s}): "
          f"{row['num_estaciones']:6.1f} "
          f"{row['num_transbordos']:10.2f} "
          f"{row['distancia_km']:9.2f} "
          f"{row['tiempo_minutos']:12.1f} "
          f"{h_label:>14} "
          f"{d_label:>9}")

df['cluster']       = labels
df['cluster_nombre'] = [nombres_cluster[l] for l in labels]

# ══════════════════════════════════════════════════════════════════
# 6. REDUCCIÓN PCA PARA VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════
print(f"\n[6] Reducción PCA (8D → 2D) para visualización...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"    Varianza explicada CP1 : {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"    Varianza explicada CP2 : {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"    Total explicado        : {sum(pca.explained_variance_ratio_)*100:.2f}%")

# ══════════════════════════════════════════════════════════════════
# 7. SILHOUETTE POR MUESTRA
# ══════════════════════════════════════════════════════════════════
sil_muestras = silhouette_samples(X_scaled, labels)

# ══════════════════════════════════════════════════════════════════
# 8. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════
print(f"\n[7] Generando visualizaciones...")

COLORES   = ['#1F3864', '#2E7D32', '#E65100', '#6A1B9A', '#00838F']
COLORES_C = [COLORES[i] for i in range(k_optimo)]

# ── 8a. Panel principal: Elbow + Silhouette + PCA + Perfiles ─────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#F8F9FA')
gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# ── Gráfico 1: Método del Codo ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ks = list(K_RANGO)
ax1.plot(ks, inercias, 'o-', color='#1F3864', linewidth=2, markersize=7)
ax1.axvline(x=k_optimo, color='#E65100', linestyle='--', linewidth=1.8,
            label=f'K óptimo = {k_optimo}')
ax1.fill_between(ks, inercias, alpha=0.08, color='#1F3864')
ax1.set_xlabel("Número de clusters (K)", fontsize=11)
ax1.set_ylabel("Inercia (WCSS)", fontsize=11)
ax1.set_title("Método del Codo", fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_facecolor('#FFFFFF')
ax1.grid(alpha=0.3)

# ── Gráfico 2: Silhouette por K ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
colores_sil = ['#E65100' if k == k_optimo else '#BDD7EE' for k in ks]
bars = ax2.bar(ks, silhouettes, color=colores_sil, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, silhouettes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)
ax2.set_xlabel("Número de clusters (K)", fontsize=11)
ax2.set_ylabel("Silhouette Score", fontsize=11)
ax2.set_title("Coeficiente de Silhouette por K", fontsize=13, fontweight='bold')
ax2.set_facecolor('#FFFFFF')
ax2.grid(axis='y', alpha=0.3)

# ── Gráfico 3: Dispersión PCA ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for c in range(k_optimo):
    mask = labels == c
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=COLORES_C[c], alpha=0.65, s=35, label=f'Cluster {c}: {nombres_cluster[c]}',
                edgecolors='white', linewidth=0.4)

# Centroides en espacio PCA
centros_pca = pca.transform(modelo.cluster_centers_)
ax3.scatter(centros_pca[:, 0], centros_pca[:, 1],
            c='black', s=160, marker='X', zorder=5, label='Centroides')
ax3.set_xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
ax3.set_ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
ax3.set_title("Clusters — Proyección PCA 2D", fontsize=13, fontweight='bold')
ax3.legend(fontsize=8.5, loc='upper right')
ax3.set_facecolor('#FFFFFF')
ax3.grid(alpha=0.2)

# ── Gráfico 4: Perfiles de clusters — Tiempo ─────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for c in range(k_optimo):
    mask = labels == c
    tiempos = df.loc[mask, 'tiempo_minutos']
    ax4.hist(tiempos, bins=18, alpha=0.65, color=COLORES_C[c],
             edgecolor='white', linewidth=0.5,
             label=f'C{c}: {nombres_cluster[c]} (μ={tiempos.mean():.0f}min)')
ax4.set_xlabel("Tiempo de viaje (min)", fontsize=11)
ax4.set_ylabel("Frecuencia", fontsize=11)
ax4.set_title("Distribución de Tiempo por Cluster", fontsize=13, fontweight='bold')
ax4.legend(fontsize=8.5)
ax4.set_facecolor('#FFFFFF')
ax4.grid(alpha=0.2)

# ── Gráfico 5: Perfiles de clusters — Estaciones ─────────────────
ax5 = fig.add_subplot(gs[1, 1])
medias_est = [df.loc[labels == c, 'num_estaciones'].mean() for c in range(k_optimo)]
medias_dis = [df.loc[labels == c, 'distancia_km'].mean() for c in range(k_optimo)]
x_pos = np.arange(k_optimo)
width = 0.38
b1 = ax5.bar(x_pos - width/2, medias_est, width, color=[COLORES_C[c] for c in range(k_optimo)],
             alpha=0.85, label='Estaciones (media)', edgecolor='white')
b2 = ax5.bar(x_pos + width/2, medias_dis, width, color=[COLORES_C[c] for c in range(k_optimo)],
             alpha=0.45, label='Distancia km (media)', edgecolor='white', hatch='//')
for b, v in zip(b1, medias_est):
    ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f'{v:.1f}',
             ha='center', fontsize=8.5)
for b, v in zip(b2, medias_dis):
    ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f'{v:.1f}',
             ha='center', fontsize=8.5)
ax5.set_xticks(x_pos)
ax5.set_xticklabels([f'C{c}\n{nombres_cluster[c]}' for c in range(k_optimo)], fontsize=9)
ax5.set_title("Estaciones y Distancia por Cluster", fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.set_facecolor('#FFFFFF')
ax5.grid(axis='y', alpha=0.2)

# ── Gráfico 6: Silhouette por muestra ────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
y_lower = 10
for c in range(k_optimo):
    mask_c    = labels == c
    sil_c     = np.sort(sil_muestras[mask_c])
    size_c    = sil_c.shape[0]
    y_upper   = y_lower + size_c
    ax6.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_c,
                      facecolor=COLORES_C[c], alpha=0.75,
                      edgecolor='white', linewidth=0.3)
    ax6.text(-0.04, y_lower + size_c/2, f'C{c}', fontsize=8,
             color=COLORES_C[c], fontweight='bold')
    y_lower = y_upper + 10

ax6.axvline(x=sil_final, color='red', linestyle='--', linewidth=1.5,
            label=f'Media: {sil_final:.3f}')
ax6.set_xlabel("Coeficiente de Silhouette", fontsize=11)
ax6.set_ylabel("Muestras por cluster", fontsize=11)
ax6.set_title("Silhouette por Muestra", fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.set_facecolor('#FFFFFF')
ax6.grid(axis='x', alpha=0.2)

fig.suptitle("K-Means Clustering — TransMilenio Bogotá\n"
             "Ref: Palma Méndez (2008), Cap. 16 — Técnicas de agrupamiento",
             fontsize=14, fontweight='bold', color='#1F3864', y=0.98)

plt.savefig("clustering_panel.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("    → clustering_panel.png")

# ── 8b. Radar de centroides ───────────────────────────────────────
from matplotlib.patches import FancyBboxPatch

fig2, axes = plt.subplots(1, k_optimo, figsize=(6*k_optimo, 5),
                           subplot_kw=dict(polar=True))
fig2.patch.set_facecolor('#F8F9FA')

# Features para radar (las más interpretables)
RADAR_FEAT = ['num_estaciones', 'num_transbordos', 'distancia_km',
              'tiempo_minutos', 'hora_periodo']
RADAR_NOM  = ['Estaciones', 'Transbordos', 'Distancia', 'Tiempo', 'Hora']
N_FEAT = len(RADAR_FEAT)
angulos = [n / float(N_FEAT) * 2 * np.pi for n in range(N_FEAT)]
angulos += angulos[:1]

for c, ax in enumerate(axes if k_optimo > 1 else [axes]):
    mask = labels == c
    vals = []
    for feat in RADAR_FEAT:
        mn = df[feat].min(); mx = df[feat].max()
        v  = df.loc[mask, feat].mean()
        vals.append((v - mn) / (mx - mn + 1e-9))
    vals += vals[:1]

    ax.plot(angulos, vals, color=COLORES_C[c], linewidth=2)
    ax.fill(angulos, vals, color=COLORES_C[c], alpha=0.30)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(RADAR_NOM, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['25%', '50%', '75%'], fontsize=7)
    ax.set_title(f"Cluster {c}\n{nombres_cluster[c]}\n"
                 f"({(labels==c).sum()} registros)",
                 size=11, fontweight='bold', color=COLORES_C[c], pad=15)
    ax.grid(color='grey', alpha=0.3)

fig2.suptitle("Perfil Radar de Clusters — K-Means TransMilenio\n"
              "Valores normalizados 0–1 por variable",
              fontsize=13, fontweight='bold', color='#1F3864')
plt.tight_layout()
plt.savefig("clustering_radar.png", dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.close()
print("    → clustering_radar.png")

# ══════════════════════════════════════════════════════════════════
# 9. GUARDAR DATASET CON CLUSTERS ASIGNADOS
# ══════════════════════════════════════════════════════════════════
df.to_csv("dataset_con_clusters.csv", index=False)
print("    → dataset_con_clusters.csv")

# ══════════════════════════════════════════════════════════════════
# 10. RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RESUMEN DEL MODELO K-MEANS")
print("=" * 65)
print(f"  Algoritmo          : K-Means (Lloyd) con K-Means++")
print(f"  K óptimo           : {k_optimo}")
print(f"  Inercia (WCSS)     : {inercia:.2f}")
print(f"  Silhouette Score   : {sil_final:.4f}  "
      f"{'(buena cohesión)' if sil_final > 0.4 else '(cohesión moderada)'}")
print(f"  Varianza PCA       : {sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"\n  Clusters descubiertos:")
for c in range(k_optimo):
    mask = labels == c
    t_med = df.loc[mask, 'tiempo_minutos'].mean()
    e_med = df.loc[mask, 'num_estaciones'].mean()
    print(f"    Cluster {c} — {nombres_cluster[c]:12s}: "
          f"{mask.sum():3d} registros | "
          f"tiempo medio {t_med:.0f} min | "
          f"{e_med:.0f} estaciones")
print("=" * 65)
