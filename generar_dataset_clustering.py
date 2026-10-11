"""
=================================================================
SISTEMA INTELIGENTE DE RUTAS - TRANSMILENIO BOGOTÁ
Dataset para Aprendizaje No Supervisado (Clustering)
=================================================================
Universidad : Corporación Universitaria Iberoamericana
Materia     : Inteligencia Artificial
Referencia  : Palma Méndez, J. T. (2008). Inteligencia artificial:
              métodos, técnicas y aplicaciones. McGraw-Hill España.
              Capítulo 16: Técnicas de agrupamiento.

Descripción:
  Genera un dataset de viajes en TransMilenio con características
  de cada trayecto. A diferencia del aprendizaje supervisado,
  NO se incluye la etiqueta de categoría — el algoritmo de
  clustering debe descubrir los grupos de forma autónoma.

Variables del dataset (todas numéricas, sin etiqueta):
  - troncal_origen_cod  : troncal de origen codificada (0-3)
  - troncal_destino_cod : troncal de destino codificada (0-3)
  - hora_periodo        : período horario (0=madrugada … 5=noche)
  - dia_tipo            : tipo de día (0=hábil, 1=sábado, 2=festivo)
  - num_estaciones      : número de estaciones en la ruta
  - num_transbordos     : transbordos necesarios (0 o 1)
  - tiempo_minutos      : tiempo real de viaje en minutos
  - distancia_km        : distancia aproximada del recorrido

Nota: tiempo_minutos y distancia_km se incluyen como features
adicionales (en supervisado era la variable objetivo implícita).
=================================================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ── Configuración de la red TransMilenio ─────────────────────────
TRONCALES = {
    "Caracas": {
        "estaciones": [
            "Portal Norte", "Toberin", "Cardio Infantil", "Mazuren",
            "Calle 100", "Calle 72", "Calle 45", "Marly",
            "Calle 26", "Av Jimenez", "Las Aguas", "Bicentenario",
            "Santa Isabel", "NQS Sur", "Portal Sur"
        ],
        "codigo": 0,
        "longitud_km": 24.0,
    },
    "Americas": {
        "estaciones": [
            "Portal Americas", "Patio Bonito", "Tintal", "Banderas",
            "Marsella", "General Santander", "Americas Carrera 30",
            "Ricaurte"
        ],
        "codigo": 1,
        "longitud_km": 13.5,
    },
    "Calle80": {
        "estaciones": [
            "Portal 80", "Quirigua", "Bolivia", "Granja Carrera 77",
            "Minuto de Dios", "Av Rojas", "El Tiempo",
            "Escuela Militar"
        ],
        "codigo": 2,
        "longitud_km": 12.0,
    },
    "Suba": {
        "estaciones": [
            "Portal Suba", "Suba Calle 100", "Niza Calle 127",
            "Pepe Sierra"
        ],
        "codigo": 3,
        "longitud_km": 8.5,
    },
}

# Velocidad promedio por troncal (km/h) — varía por troncal
VELOCIDAD_BASE = {"Caracas": 28, "Americas": 30, "Calle80": 32, "Suba": 35}

# Factor por hora del día (impacto en velocidad)
FACTOR_HORA = {0: 1.10, 1: 0.70, 2: 0.95, 3: 0.85, 4: 0.72, 5: 1.05}
FACTOR_DIA  = {0: 1.00, 1: 1.15, 2: 1.25}

N = 500   # registros totales

registros = []

for _ in range(N):
    # Seleccionar troncales
    troncal_orig_nom = np.random.choice(list(TRONCALES.keys()))
    troncal_dest_nom = np.random.choice(list(TRONCALES.keys()))

    t_orig = TRONCALES[troncal_orig_nom]
    t_dest = TRONCALES[troncal_dest_nom]

    # Estación origen y destino
    est_orig = np.random.choice(t_orig["estaciones"])
    est_dest = np.random.choice(t_dest["estaciones"])
    if est_orig == est_dest:
        continue

    # Parámetros del viaje
    hora_periodo = np.random.randint(0, 6)
    dia_tipo     = np.random.choice([0, 1, 2], p=[0.71, 0.19, 0.10])

    # Número de estaciones
    if troncal_orig_nom == troncal_dest_nom:
        num_estaciones = abs(
            t_orig["estaciones"].index(est_orig) -
            t_orig["estaciones"].index(est_dest)
        ) + 1
        num_transbordos = 0
    else:
        # Ruta inter-troncal: pasa por estación de conexión
        num_estaciones = np.random.randint(6, 20)
        num_transbordos = 1

    # Distancia aproximada
    km_por_estacion = (t_orig["longitud_km"] / len(t_orig["estaciones"]))
    distancia_km = round(num_estaciones * km_por_estacion *
                         np.random.uniform(0.9, 1.1), 2)

    # Velocidad efectiva con factores
    vel_base = (VELOCIDAD_BASE[troncal_orig_nom] +
                VELOCIDAD_BASE[troncal_dest_nom]) / 2
    vel_efectiva = (vel_base
                    * FACTOR_HORA[hora_periodo]
                    * FACTOR_DIA[dia_tipo]
                    * np.random.uniform(0.92, 1.08))

    # Tiempo en minutos
    tiempo_base = (distancia_km / vel_efectiva) * 60
    if num_transbordos > 0:
        tiempo_base += 5   # tiempo de transbordo
    tiempo_minutos = round(tiempo_base + np.random.uniform(-2, 2), 1)
    tiempo_minutos = max(3.0, tiempo_minutos)

    registros.append({
        "estacion_origen":     est_orig,
        "estacion_destino":    est_dest,
        "troncal_origen":      troncal_orig_nom,
        "troncal_destino":     troncal_dest_nom,
        "troncal_origen_cod":  t_orig["codigo"],
        "troncal_destino_cod": t_dest["codigo"],
        "hora_periodo":        hora_periodo,
        "dia_tipo":            dia_tipo,
        "num_estaciones":      num_estaciones,
        "num_transbordos":     num_transbordos,
        "distancia_km":        distancia_km,
        "tiempo_minutos":      tiempo_minutos,
    })

df = pd.DataFrame(registros).reset_index(drop=True)

print("=" * 60)
print("  DATASET CLUSTERING — TRANSMILENIO BOGOTÁ")
print("=" * 60)
print(f"  Registros generados : {len(df)}")
print(f"  Variables           : {df.shape[1]}")
print(f"\n  Estadísticas de tiempo_minutos:")
print(df['tiempo_minutos'].describe().round(2).to_string())
print(f"\n  Distribución hora_periodo:")
for h, n in df['hora_periodo'].value_counts().sort_index().items():
    labels = ["Madrugada","Mañana pico","Mañana","Tarde","Tarde pico","Noche"]
    print(f"    {h} ({labels[h]:12s}): {n:3d}")
print(f"\n  Distribución troncal_origen:")
print(df['troncal_origen'].value_counts().to_string())

df.to_csv("dataset_clustering_transmilenio.csv", index=False)
print(f"\n  ✅ Guardado: dataset_clustering_transmilenio.csv")
print("=" * 60)
