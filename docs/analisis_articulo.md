# Análisis del artículo DeepACO (TSP)

## 1. Problema

**Problema del Agente Viajero (TSP):** dadas *n* ciudades, encontrar el recorrido más corto que visite cada una exactamente una vez y regrese al punto de partida.

## 2. Formulación

**Datos.** Coordenadas de *n* ciudades muestreadas uniformemente en el cuadrado unitario [0, 1]² (Apéndice D.1) y distancias euclidianas d<sub>ij</sub>.

**Variables de decisión.** x<sub>ij</sub> ∈ {0, 1}, igual a 1 si el recorrido va de la ciudad *i* a la *j*. En ACO la solución se representa directamente como una permutación s = (s<sub>1</sub>, …, s<sub>n</sub>).

**Función objetivo (minimizar).**

f(s) = Σ<sub>t=1..n</sub> d(s<sub>t</sub>, s<sub>t+1</sub>), con s<sub>n+1</sub> = s<sub>1</sub>

**Restricciones.** Cada ciudad se visita exactamente una vez y el recorrido es un solo ciclo (sin subtours). En ACO se cumplen por construcción: cada hormiga enmascara las ciudades ya visitadas.

**Objetivo de entrenamiento de la red (Ec. 4, con W = 0 en la versión sin búsqueda local).**

min<sub>θ</sub> E<sub>s ~ P<sub>η<sub>θ</sub></sub></sub> [ f(s) ]

estimado con REINFORCE usando como línea base el costo promedio de las hormigas (Ec. 5).

## 3. Algoritmo bioinspirado

**ACO.** Una hormiga en la ciudad *i* elige la siguiente ciudad *j* (no visitada) con probabilidad

P(j | i) = τ<sub>ij</sub><sup>α</sup> η<sub>ij</sub><sup>β</sup> / Σ<sub>l</sub> τ<sub>il</sub><sup>α</sup> η<sub>il</sub><sup>β</sup>

Tras cada iteración la feromona se evapora (τ ← decay · τ) y cada arista usada recibe 1/f(s). Variantes evaluadas:

- **Ant System (AS):** depositan todas las hormigas.
- **Elitist AS:** solo deposita la mejor hormiga de la iteración.
- **MAX-MIN AS:** la feromona se acota a [τ<sub>min</sub>, τ<sub>max</sub>], con τ<sub>max</sub> = n / mejor costo.

**DeepACO.** La heurística η<sub>ij</sub> = 1/d<sub>ij</sub> se sustituye por η<sub>θ</sub> producida por una red neuronal de grafos (12 capas, 32 unidades, paso de mensajes anisotrópico con compuertas de arista y decodificador MLP de 3 capas con salida sigmoide; Apéndice A).

**Esparsificación.** Cada ciudad solo se conecta con sus *k* vecinos más cercanos, tanto en ACO como en DeepACO.

## 4. Hiperparámetros: artículo vs código

| Parámetro | Artículo | Código (`tsp/`) |
|---|---|---|
| α, β | 1, 1 (Sección 3) | 1, 1 |
| k vecinos | 10 (TSP20), 20 (TSP100), 50 (TSP500) | igual |
| Hormigas | **no se reporta** | 20 (TSP20, TSP100), 50 (TSP500) |
| Iteraciones T | **no se reporta**; la Fig. 2 usa "evaluaciones" en el eje x | test.ipynb: T ∈ {1, 10, 20, 30, 40, 50, 100} (solo TSP20) |
| Evaporación (decay) | **no se reporta** (solo se barre en la Fig. 4) | 0.9 |
| τ<sub>min</sub> en MMAS | **no se reporta** | 0.1 |
| Entrenamiento | 640 instancias; 1, 2 y 8 min (Tabla 7) | 5 épocas × 128 pasos = 640; lr = 3·10⁻⁴, AdamW |
| Instancias de prueba | 100 por tamaño | archivos con 1280 (TSP20, TSP100) y 128 (TSP500) |
| Corridas | 3 (Fig. 2) | 1 (test.ipynb, semilla 12345) |
| Hardware | RTX 3090 + Xeon de 48 núcleos | — |

**Resultados de referencia en el repositorio de los autores** (test.ipynb, TSP20, AS, 20 hormigas, 1280 instancias, una corrida):

| T | 1 | 10 | 20 | 30 | 40 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| ACO | 5.255 | 4.207 | 4.042 | 3.949 | 3.911 | 3.889 | 3.835 |
| DeepACO | 3.932 | 3.823 | 3.817 | 3.812 | 3.811 | 3.811 | 3.809 |

Son los únicos valores numéricos que los autores publican para la Fig. 2; los demás solo aparecen como curvas.

## 5. Protocolo de la réplica

- Instancias: las primeras 100 de los archivos de prueba de los autores (el artículo reporta 100 instancias de prueba).
- Modelos: pesos preentrenados de los autores (`deepaco/pretrained/tsp`).
- 30 corridas independientes con semillas 0–29; ACO y DeepACO usan la misma semilla, lo que permite una comparación pareada.
- Métrica por corrida: promedio, sobre las 100 instancias, del mejor costo encontrado hasta la iteración T.
- Estadística: media, mediana y desviación estándar sobre las 30 corridas; Wilcoxon de rangos con signo pareado (ACO vs DeepACO) y de una muestra (réplica vs valor reportado).

## 6. Hallazgos de replicabilidad (bitácora)

1. **Hiperparámetros omitidos en el artículo.** Número de hormigas, número de iteraciones, tasa de evaporación y τ<sub>min</sub> de MMAS no aparecen en el texto; se tomaron del código.
2. **Número de instancias de prueba.** El artículo dice 100; los archivos del repositorio tienen 1280 (TSP20, TSP100) y 128 (TSP500), y el notebook de prueba evalúa sobre todos.
3. **Solo hay un ejemplo de prueba.** El notebook `test.ipynb` solo muestra TSP20 con Ant System; para Elitist AS, MMAS, TSP100 y TSP500 no hay configuración de prueba publicada.
4. **Dependencias obsoletas.** El código pide Python 3.8, PyTorch 1.7 y PyG 2.0.4, que ya no se pueden instalar en entornos actuales como Colab. Funciona con PyTorch 2.14 y PyG 2.8 con un solo cambio:
5. **Validación de probabilidades.** A partir de PyTorch 1.8, `torch.distributions.Categorical` valida que las probabilidades formen un simplex. En TSP500 la heurística aprendida produce valores muy pequeños (~10⁻¹²) y la validación aborta la ejecución. Se agregó `validate_args=False` en `deepaco/tsp/aco.py`, lo que reproduce el comportamiento de PyTorch 1.7. Los propios autores ya usan ese argumento en `tsp_nls/aco.py`, pero no en `tsp/aco.py`.
6. **Costo computacional.** Los autores usan una GPU RTX 3090 y reportan 3 corridas; la actividad pide 30. En CPU, TSP500 con 100 instancias y T = 100 tarda ~1.5 h por método y semilla, por lo que requiere GPU o menos instancias.

Código de los autores usado: commit `9a756a3` (29 de septiembre de 2024).
