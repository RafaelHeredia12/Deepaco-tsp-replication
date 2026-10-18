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

- Instancias: las primeras 100 de los archivos de prueba de los autores (el artículo reporta 100 instancias de prueba). En TSP500, las primeras 20.
- Modelos: pesos preentrenados de los autores (`deepaco/pretrained/tsp`).
- 30 corridas independientes con semillas 0–29; ACO y DeepACO usan la misma semilla, lo que permite una comparación pareada.
- Métrica por corrida: promedio, sobre las 100 instancias, del mejor costo encontrado hasta la iteración T.
- Estadística: media, mediana y desviación estándar sobre las 30 corridas; Wilcoxon de rangos con signo pareado (ACO vs DeepACO) y de una muestra (réplica vs valor reportado).

## 6. Hallazgos de replicabilidad (bitácora)

1. **Hiperparámetros omitidos en el artículo.** Número de hormigas, número de iteraciones, tasa de evaporación y τ<sub>min</sub> de MMAS no aparecen en el texto; se tomaron del código.
2. **Número de instancias de prueba.** El artículo dice 100; los archivos del repositorio tienen 1280 (TSP20, TSP100) y 128 (TSP500), y el notebook de prueba evalúa sobre todos.
3. **Solo hay un ejemplo de prueba.** El notebook `test.ipynb` solo muestra TSP20 con Ant System; para Elitist AS, MMAS, TSP100 y TSP500 no hay configuración de prueba publicada.
4. **Dependencias obsoletas.** El código pide Python 3.8, PyTorch 1.7 y PyG 2.0.4, que ya no se pueden instalar en entornos actuales como Colab. Funciona con PyTorch 2.14 y PyG 2.8 con un solo cambio (punto 5).
5. **Validación de probabilidades.** A partir de PyTorch 1.8, `torch.distributions.Categorical` valida que las probabilidades formen un simplex. En TSP500 la heurística aprendida produce valores muy pequeños (~10⁻¹²) y la validación aborta la ejecución. Se agregó `validate_args=False` en `deepaco/tsp/aco.py`, lo que reproduce el comportamiento de PyTorch 1.7. Los propios autores ya usan ese argumento en `tsp_nls/aco.py`, pero no en `tsp/aco.py`.
6. **Costo computacional.** Los autores usan una GPU RTX 3090 y reportan 3 corridas; la actividad pide 30. En CPU, TSP500 con 100 instancias y T = 100 tarda ~1.5 h por método y semilla, por lo que requiere GPU o menos instancias.
7. **Protocolo reducido en TSP500.** Por costo computacional, TSP500 se corrió solo con Ant System, 20 instancias (en vez de 100) y hasta T = 50, manteniendo las 30 semillas.
8. **GPU no siempre es más rápida.** En TSP100, el código de los autores corre unas 3.5 veces más lento en una RTX 3070 que en CPU: cada paso de construcción de las hormigas es una operación muy pequeña y domina el costo de enviarla a la GPU. La GPU solo conviene en TSP500.

Código de los autores usado: commit `9a756a3` (29 de septiembre de 2024).

## 7. Mejora propuesta: hibridación con 2-opt

**Idea.** Cada ruta construida por una hormiga se refina con búsqueda local 2-opt antes de evaluarla y depositar feromona (esquema clásico ACO + búsqueda local; Dorigo & Stützle). 2-opt elimina cruces: si los arcos (a, b) y (c, d) se cruzan, se reemplazan por (a, c) y (b, d) invirtiendo el tramo intermedio, lo que en el plano euclidiano siempre acorta la ruta.

**Por qué esta mejora.** En un piloto (TSP100, 10 instancias, 2 semillas) se compararon tres candidatas frente a la réplica de DeepACO (AS):

| Configuración | Longitud media | vs. réplica |
|---|---|---|
| DeepACO + AS (réplica) | 8.130 | — |
| DeepACO + Ant Colony System | 8.097 | −0.4 % |
| DeepACO con heurística η<sub>θ</sub>·(1/d) | 8.115 | −0.2 % |
| DeepACO + AS + 2-opt | 7.759 | −4.6 % |
| DeepACO + ACS + 2-opt | 7.739 | −4.8 % |
| ACO clásico + 2-opt (control) | 7.727 | −5.0 % |

Solo la hibridación con 2-opt dio una mejora importante, sin aumentar de forma notable el tiempo. El piloto también sugiere que, con 2-opt, el ACO clásico alcanza a DeepACO. Esto es coherente con la limitación que señalan los autores: la heurística aprendida sin búsqueda local es muy "agresiva" (Apéndice C.6) y reduce la exploración, por lo que para usar búsqueda local entrenan un modelo distinto (W = 9 en la Ec. 4). El experimento completo (30 semillas, 100 instancias) lo confirmó (sección 9).

**Relación con el artículo.** Los autores proponen NLS, una búsqueda local que alterna 2-opt con perturbaciones guiadas por la red, con un modelo reentrenado para ese fin. La mejora aquí es más simple: un 2-opt estándar sobre el modelo básico, sin reentrenar.

**Parámetros de ACS.** q<sub>0</sub> = 0.9, ρ = 0.1, ξ = 0.1, τ<sub>0</sub> = 1/(n·L̄) con L̄ la longitud media de la primera iteración (valores estándar de Dorigo & Gambardella, 1997).

## 8. Resultados de la réplica

Longitud media del tour al final de la búsqueda, 30 corridas (± desviación estándar). Wilcoxon pareado ACO vs DeepACO: p < 10⁻⁸ en todas las configuraciones; DeepACO gana en las 30 corridas de cada una.

| Tamaño | Variante | T | ACO clásico | DeepACO | DeepACO mejor por |
|---|---|---|---|---|---|
| 20 | AS | 100 | 3.829 ± 0.002 | 3.811 ± 0.001 | 0.5 % |
| 20 | Elitist | 100 | 3.817 ± 0.002 | 3.808 ± 0.001 | 0.2 % |
| 20 | MMAS | 100 | 4.281 ± 0.013 | 3.811 ± 0.001 | 11.0 % |
| 100 | AS | 100 | 9.664 ± 0.023 | 8.234 ± 0.007 | 14.8 % |
| 100 | Elitist | 100 | 9.439 ± 0.028 | 8.129 ± 0.007 | 13.9 % |
| 100 | MMAS | 100 | 9.979 ± 0.016 | 8.276 ± 0.006 | 17.1 % |
| 500 | AS | 50 | 26.531 ± 0.066 | 18.889 ± 0.023 | 28.8 % |

**Comparación con los valores publicados (TSP20, AS).**

| T | 1 | 10 | 20 | 30 | 40 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| ACO réplica | 5.218 | 4.219 | 4.029 | 3.947 | 3.902 | 3.875 | 3.829 |
| ACO artículo | 5.255 | 4.207 | 4.042 | 3.949 | 3.911 | 3.889 | 3.835 |
| DeepACO réplica | 3.928 | 3.826 | 3.819 | 3.816 | 3.815 | 3.813 | 3.811 |
| DeepACO artículo | 3.932 | 3.823 | 3.817 | 3.812 | 3.811 | 3.811 | 3.809 |

La diferencia máxima es 0.7 % (ACO, T = 1) y en T = 100 es menor a 0.2 %. El Wilcoxon de una muestra contra el valor publicado rechaza la igualdad (p < 10⁻⁸) porque con 30 corridas la dispersión es muy pequeña, pero la diferencia práctica es despreciable y además se esperaba: los autores reportan una sola corrida sobre 1280 instancias, y la réplica usa 30 corridas sobre las primeras 100. Para TSP100 y TSP500 el artículo solo muestra curvas, sin valores numéricos; la réplica reproduce su forma (DeepACO por debajo de ACO en todas las iteraciones y variantes).

**Observaciones.**
- La ventaja de DeepACO crece con el tamaño: de 0.2–11 % en TSP20 a 14–17 % en TSP100 y 29 % en TSP500.
- Con una sola iteración, DeepACO (9.02 en TSP100) ya supera a ACO clásico tras 100 iteraciones (9.44–9.98).
- DeepACO es mucho más estable: su desviación estándar es 3 a 4 veces menor.
- DeepACO es casi insensible a la variante de ACO (3.808–3.811 en TSP20), mientras que ACO clásico con MMAS converge mucho más lento en 100 iteraciones (4.28). Esto coincide con la afirmación del artículo de que DeepACO es más robusto a la configuración del algoritmo.

## 9. Resultados de la mejora (TSP100, Ant System, 30 corridas)

| Método | Longitud media | vs. réplica de DeepACO | Wilcoxon vs. réplica |
|---|---|---|---|
| ACO clásico | 9.664 ± 0.023 | 17.4 % peor | p < 10⁻⁸ |
| DeepACO (réplica) | 8.234 ± 0.007 | — | — |
| **DeepACO + 2-opt** | **7.817 ± 0.002** | **5.1 % mejor** | p < 10⁻⁸ |
| DeepACO + ACS + 2-opt | 7.791 ± 0.002 | 5.4 % mejor | p < 10⁻⁸ |
| ACO clásico + 2-opt | 7.785 ± 0.001 | 5.5 % mejor | p < 10⁻⁸ |

- **La mejora funciona:** el 2-opt reduce la longitud un 5.1 % y deja a DeepACO a 0.7 % del mejor valor del artículo en TSP100 (7.763, Tabla 3, con NLS y un modelo reentrenado), sin reentrenar la red.
- **Con 2-opt, el ACO clásico alcanza a DeepACO.** ACO + 2-opt supera a DeepACO + 2-opt en las 30 corridas (Wilcoxon p < 10⁻⁸, diferencia de 0.4 %) y en 93 de 100 instancias. DeepACO + 2-opt va adelante en la primera iteración (7.987 vs 8.024), pero ACO + 2-opt lo alcanza hacia T = 10 y lo supera después.
- **ACS recupera exploración:** DeepACO + ACS + 2-opt mejora a DeepACO + 2-opt en las 30 corridas y queda a 0.07 % de ACO + 2-opt.

## 10. Conclusiones

**Comportamiento y efectividad del algoritmo.** DeepACO cumple lo que promete: con el mismo número de hormigas e iteraciones encuentra rutas entre 14 % y 29 % más cortas que el ACO clásico en problemas medianos y grandes, converge desde la primera iteración y es menos sensible a la variante de ACO. La ventaja es pequeña en TSP20, donde ambos métodos llegan cerca del óptimo.

**La heurística aprendida acelera, pero reduce la exploración.** Al agregar búsqueda local 2-opt, lo que más aporta es la diversidad de rutas que la búsqueda local pule, y ahí la heurística clásica 1/d, más "suave", termina ligeramente por delante. Esto confirma con datos propios la limitación que reconocen los autores (heurística "agresiva", Apéndice C.6) y explica por qué entrenan un modelo distinto cuando usan búsqueda local (W = 9). Ant Colony System atenúa el problema.

**Replicabilidad.** Alta. El código, los datos y los modelos preentrenados son públicos, y los únicos valores numéricos publicados para la Fig. 2 se reprodujeron con diferencias menores a 0.2 %. Las dificultades fueron de documentación y de entorno: hiperparámetros omitidos en el texto (hormigas, iteraciones, evaporación), una discrepancia en el número de instancias de prueba, dependencias obsoletas que obligaron a un cambio de una línea, y solo un ejemplo de configuración de prueba publicado.

**Dificultades encontradas.** El costo computacional (el artículo usa 3 corridas en una RTX 3090; aquí se pidieron 30), que obligó a reducir el protocolo de TSP500 y a paralelizar en varias terminales; y que la GPU resultó más lenta que la CPU en TSP100 con este código.
