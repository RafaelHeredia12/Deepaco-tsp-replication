# Réplica de DeepACO en el Problema del Agente Viajero (TSP)

Réplica y extensión del artículo:

> Ye, H., Wang, J., Cao, Z., Liang, H., & Li, Y. (2023). DeepACO: Neural-enhanced ant systems for combinatorial optimization. *Advances in Neural Information Processing Systems, 36*, 43706–43728. https://arxiv.org/abs/2309.14032

Código original de los autores: https://github.com/henry-yeh/DeepACO (licencia MIT).

**DeepACO** combina la Optimización por Colonia de Hormigas (ACO) con una red neuronal de grafos que aprende, por aprendizaje por refuerzo, la heurística η que guía a las hormigas, en lugar de usar la heurística diseñada a mano (η = 1/distancia).

## Alcance de la réplica

Replicamos la evaluación del artículo en TSP:

Se replicó completo el **panel TSP de la Fig. 2** (ACO vs DeepACO con Ant System, Elitist AS y MAX-MIN AS):

| Tamaño | Variantes | Instancias | Iteraciones T | Corridas |
|---|---|---|---|---|
| TSP20 | AS, Elitist, MMAS | 100 | 100 | 30 |
| TSP100 | AS, Elitist, MMAS | 100 | 100 | 30 |
| TSP500 | AS | 20 (protocolo reducido por costo) | 50 | 30 |

Protocolo: **30 corridas independientes** (semillas 0–29) por configuración, frente a 3 en el artículo; se reportan media, mediana y desviación estándar, y se compara con la prueba de Wilcoxon de rangos con signo. El análisis completo está en [`docs/analisis_articulo.md`](docs/analisis_articulo.md).

**Fuera de alcance:** la Fig. 4 (sensibilidad a hiperparámetros), la Fig. 5 (TSPLIB) y la Tabla 1 (comparación con otros métodos neuronales con búsqueda local NLS). Se priorizó replicar completa la Fig. 2 con 30 corridas por configuración.

## Resultados

Longitud media del tour al final de la búsqueda (30 corridas; ± desviación estándar):

| | ACO clásico | DeepACO | DeepACO mejor por |
|---|---|---|---|
| TSP20 · AS | 3.829 ± 0.002 | 3.811 ± 0.001 | 0.5 % |
| TSP20 · Elitist | 3.817 ± 0.002 | 3.808 ± 0.001 | 0.2 % |
| TSP20 · MMAS | 4.281 ± 0.013 | 3.811 ± 0.001 | 11.0 % |
| TSP100 · AS | 9.664 ± 0.023 | 8.234 ± 0.007 | 14.8 % |
| TSP100 · Elitist | 9.439 ± 0.028 | 8.129 ± 0.007 | 13.9 % |
| TSP100 · MMAS | 9.979 ± 0.016 | 8.276 ± 0.006 | 17.1 % |
| TSP500 · AS | 26.531 ± 0.066 | 18.889 ± 0.023 | 28.8 % |

DeepACO supera a ACO en las 30 corridas de todas las configuraciones (Wilcoxon pareado, p < 10⁻⁸). En TSP20, los únicos valores numéricos que publican los autores (ACO 3.835, DeepACO 3.809 en T = 100) quedan a menos de 0.2 % de la réplica (3.829 y 3.811).

**Mejora (TSP100, Ant System):**

| Método | Longitud media | vs. réplica de DeepACO |
|---|---|---|
| DeepACO (réplica) | 8.234 ± 0.007 | — |
| **DeepACO + 2-opt** | **7.817 ± 0.002** | **5.1 % mejor** |
| DeepACO + ACS + 2-opt | 7.791 ± 0.002 | 5.4 % mejor |
| ACO clásico + 2-opt (control) | 7.785 ± 0.001 | 5.5 % mejor |

Con 2-opt, DeepACO arranca mejor pero el ACO clásico lo alcanza hacia la iteración 10 y termina ligeramente por delante (30 de 30 corridas). Las figuras están en `results/figuras/` y se regeneran con `python analysis/figuras_poster.py`.

## Mejora propuesta: hibridación con búsqueda local 2-opt

Cada ruta que construye una hormiga se mejora con **2-opt** (se buscan dos tramos que se cruzan y se "descruzan", hasta que no queden cruces) **antes** de calcular su costo y depositar feromona. Así la colonia aprende de rutas ya pulidas.

| Método (`run_mejora.py`) | Qué es |
|---|---|
| `deepaco_2opt` | DeepACO (Ant System) + 2-opt — **mejora principal** |
| `deepaco_acs_2opt` | DeepACO con Ant Colony System (Dorigo & Gambardella, 1997) + 2-opt |
| `aco_2opt` | ACO clásico (η = 1/d) + 2-opt — control |
| `deepaco_acs` | DeepACO con Ant Colony System, sin 2-opt — ablación (opcional) |

Se usa el mismo protocolo que la réplica (mismas instancias, semillas y puntos de control), así que las comparaciones contra `deepaco` son pareadas. El 2-opt es la implementación de los autores (`deepaco/tsp_nls/two_opt.py`); la clase ACS y la hibridación están en `experiments/variants.py`.

Los autores evalúan una idea relacionada (búsqueda local guiada por la red, NLS) pero con un modelo reentrenado específicamente para ello. Aquí se prueba qué ocurre al agregar un 2-opt estándar a la configuración básica que se replicó.

## Estructura

```
deepaco/                 Código original de los autores (solo TSP), datos y modelos preentrenados
  tsp/                   DeepACO básico (sin búsqueda local)
  tsp_nls/               DeepACO con búsqueda local guiada por la red (NLS)
  data/tsp/              Instancias de prueba y validación fijas
  pretrained/            Pesos entrenados por los autores
  LICENSE                Licencia MIT original
experiments/run_tsp.py   Réplica: ACO vs DeepACO con 30 semillas
experiments/run_mejora.py Mejora: hibridación con 2-opt (mismo protocolo)
experiments/variants.py  Ant Colony System e hibridación con 2-opt
analysis/stats.py        Media, mediana, d.e., Wilcoxon y curvas de evolución
analysis/figuras_poster.py Figuras y tabla resumen para el póster
docs/                    Análisis del artículo, hallazgos de replicabilidad y conclusiones
results/                 CSV de todas las corridas (réplica y mejora)
results/figuras/         Figuras del póster y tabla_resumen.csv
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

**2. Correr la mejora** (mismas opciones `--size`, `--runs`, `--first-seed`, `--instances`, `--out`):

```bash
python experiments/run_mejora.py --size 100 --runs 30 --out results/mejora_tsp100.csv
```

Usa `--methods` para elegir qué configuraciones correr (por defecto `deepaco_2opt deepaco_acs_2opt aco_2opt`). En TSP20 todas las variantes con 2-opt alcanzan el óptimo, así que la comparación útil es en TSP100.

**3. Analizar los resultados.** Acepta uno o varios CSV (o comodines), por ejemplo las dos mitades de semillas, o la réplica junto con la mejora:

```bash
python analysis/stats.py results/tsp20_as.csv --reported aco=3.835 deepaco=3.809
python analysis/stats.py "results/tsp100_as_*.csv"
python analysis/stats.py "results/tsp100_as_*.csv" "results/mejora_tsp100*.csv" --name mejora_tsp100
```

Imprime la tabla de media, mediana y desviación estándar, la prueba de Wilcoxon pareada (ACO vs DeepACO y cada método contra la réplica de DeepACO), la comparación contra los valores reportados, y guarda `*_resumen.csv` y `*_curva.png`.

**4. Figuras del póster:**

```bash
python analysis/figuras_poster.py
```

Lee todos los CSV de `results/` y genera en `results/figuras/` la réplica contra el artículo (TSP20), la ventaja de DeepACO por variante y tamaño, el efecto del 2-opt y su evolución, además de `tabla_resumen.csv`.

### Tiempos medidos

Por método y semilla, 100 instancias, T = 100, un hilo por proceso:

| Tamaño | CPU | GPU (RTX 3070) |
|---|---|---|
| TSP20 | ~30 s | — |
| TSP100 | ~4–6 min | ~13 min (más lento: los pasos de cada hormiga son cálculos muy pequeños) |
| TSP500 | inviable | ~8 min con 20 instancias y T = 50 |

Para TSP100 conviene CPU y correr varias terminales en paralelo, cada una con un rango de semillas (`--first-seed`) y su propio `--out`, poniendo antes en cada terminal `$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1` (PowerShell) para que no compitan por los núcleos. `stats.py` junta después los archivos.

## Créditos

El algoritmo, el código en `deepaco/`, los datos y los modelos preentrenados son de Ye et al. (2023) y se distribuyen bajo su licencia MIT. Los scripts de experimentación y análisis, y la documentación, son trabajo de este equipo.

**Equipo:** Rafael Santamaria Heredia y Grethel Hernandez Olivares — Tecnológico de Monterrey, Campus Guadalajara.
