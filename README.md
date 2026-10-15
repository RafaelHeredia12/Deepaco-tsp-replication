# Réplica de DeepACO en el Problema del Agente Viajero (TSP)

Réplica y extensión del artículo:

> Ye, H., Wang, J., Cao, Z., Liang, H., & Li, Y. (2023). DeepACO: Neural-enhanced ant systems for combinatorial optimization. *Advances in Neural Information Processing Systems, 36*, 43706–43728. https://arxiv.org/abs/2309.14032

Código original de los autores: https://github.com/henry-yeh/DeepACO (licencia MIT).

**DeepACO** combina la Optimización por Colonia de Hormigas (ACO) con una red neuronal de grafos que aprende, por aprendizaje por refuerzo, la heurística η que guía a las hormigas, en lugar de usar la heurística diseñada a mano (η = 1/distancia).

## Alcance de la réplica

Replicamos la evaluación del artículo en TSP:

| Experimento del artículo | Qué compara | Estado |
|---|---|---|
| Fig. 2 (panel TSP) | ACO vs DeepACO con Ant System, Elitist AS y MAX-MIN AS | Script listo (`experiments/run_tsp.py`) |
| Fig. 4 | Sensibilidad a α y a la evaporación (TSP100) | Pendiente |
| Fig. 5 | Generalización a instancias reales de TSPLIB | Pendiente |
| Tabla 1 (filas ACO y DeepACO) | TSP500 y TSP1000 con búsqueda local guiada (NLS) | Pendiente |

Protocolo: **30 corridas independientes** (semillas 0–29) por configuración; se reportan media, mediana y desviación estándar, y se compara con la prueba de Wilcoxon. El análisis completo del artículo está en [`docs/analisis_articulo.md`](docs/analisis_articulo.md).

## Estructura

```
deepaco/                 Código original de los autores (solo TSP), datos y modelos preentrenados
  tsp/                   DeepACO básico (sin búsqueda local)
  tsp_nls/               DeepACO con búsqueda local guiada por la red (NLS)
  data/tsp/              Instancias de prueba y validación fijas
  pretrained/            Pesos entrenados por los autores
  LICENSE                Licencia MIT original
experiments/run_tsp.py   Experimento ACO vs DeepACO con 30 semillas
analysis/stats.py        Media, mediana, d.e., Wilcoxon y curvas de evolución
docs/                    Análisis del artículo y hallazgos de replicabilidad
results/                 Salidas de los experimentos (CSV y gráficas)
```

Modificaciones al código original: una sola línea en `deepaco/tsp/aco.py` (`Categorical(..., validate_args=False)`) para que corra con PyTorch moderno. Está explicada en `docs/analisis_articulo.md`.

## Instalación

Probado con Python 3.13, PyTorch 2.14 y PyTorch Geometric 2.8 (los autores usaron Python 3.8 y PyTorch 1.7, que ya no se pueden instalar en Colab).

```bash
git clone https://github.com/RafaelHeredia12/Deepaco-tsp-replication.git
cd Deepaco-tsp-replication
pip install -r requirements.txt
```

En Google Colab basta con `!pip install torch_geometric` (PyTorch ya viene instalado). Activa la GPU en *Entorno de ejecución → Cambiar tipo de entorno* para los tamaños grandes.

## Uso

**1. Correr el experimento** (ACO y DeepACO, misma semilla para ambos):

```bash
python experiments/run_tsp.py --size 20 --variant as --runs 30 --instances 100
```

| Opción | Valores | Por defecto |
|---|---|---|
| `--size` | 20, 100, 500 | obligatorio |
| `--variant` | `as` (Ant System), `elitist`, `mmas` (MAX-MIN) | `as` |
| `--runs` | número de corridas independientes | 30 |
| `--first-seed` | semilla inicial, para repartir corridas en varias sesiones | 0 |
| `--instances` | instancias de prueba por corrida | 100 |
| `--checkpoints` | iteraciones T donde se registra el mejor costo | 1 10 20 30 40 50 100 |
| `--n-ants`, `--k-sparse`, `--decay`, `--alpha`, `--beta` | hiperparámetros de ACO | los del código original |
| `--device` | `cuda` o `cpu` | `cuda` si hay GPU |

Los resultados se agregan a `results/tsp{size}_{variant}.csv`, así que se pueden correr las semillas por partes (por ejemplo `--first-seed 0 --runs 15` y luego `--first-seed 15 --runs 15`).

**2. Analizar los resultados:**

```bash
python analysis/stats.py results/tsp20_as.csv --reported aco=3.835 deepaco=3.809
```

Imprime la tabla de media, mediana y desviación estándar, la prueba de Wilcoxon pareada ACO vs DeepACO, la comparación contra los valores reportados, y guarda `*_resumen.csv` y `*_curva.png`.

### Tiempos aproximados (CPU de 2 núcleos, 100 instancias, T = 100)

| Tamaño | Por método y semilla | 30 semillas × 2 métodos |
|---|---|---|
| TSP20 | ~30 s | ~30 min |
| TSP100 | ~3 min | ~3.5 h |
| TSP500 | ~1.5 h | inviable en CPU: usar GPU o menos instancias |

## Créditos

El algoritmo, el código en `deepaco/`, los datos y los modelos preentrenados son de Ye et al. (2023) y se distribuyen bajo su licencia MIT. Los scripts de experimentación y análisis, y la documentación, son trabajo de este equipo.

**Equipo:** Rafael Santamaria Heredia y Grethel Hernandez Olivares — Tecnológico de Monterrey, Campus Guadalajara.
