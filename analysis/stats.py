"""
Estadísticas a partir de los CSV de experiments/run_tsp.py y experiments/run_mejora.py.

Acepta uno o varios CSV (por ejemplo las dos mitades de semillas de una variante, o la
réplica junto con la mejora) y los junta antes de analizar. Para cada método calcula, sobre
las corridas independientes, la media, mediana y desviación estándar del costo promedio
por corrida. Después compara:
  * ACO vs DeepACO (Wilcoxon pareado por semilla), si ambos están.
  * Cada método contra la línea base --baseline (por defecto, la réplica de DeepACO).
  * Réplica vs valor reportado por los autores (Wilcoxon de una muestra), si se da --reported.
Además dibuja las curvas de evolución (costo vs número de evaluaciones), como la Fig. 2.

Ejemplos:
    python analysis/stats.py results/tsp100_as_a.csv results/tsp100_as_b.csv
    python analysis/stats.py results/tsp20_as.csv --reported aco=3.835 deepaco=3.809
    python analysis/stats.py results/tsp100_as_*.csv results/mejora_tsp100*.csv --name mejora_tsp100
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

LABELS = {
    "aco": "ACO clásico",
    "deepaco": "DeepACO (réplica)",
    "aco_2opt": "ACO clásico + 2-opt",
    "deepaco_2opt": "DeepACO + 2-opt",
    "deepaco_acs": "DeepACO + ACS",
    "deepaco_acs_2opt": "DeepACO + ACS + 2-opt",
}
COLORS = {
    "aco": "#9ca3af", "deepaco": "#c2410c",
    "aco_2opt": "#4b5563", "deepaco_2opt": "#1d4ed8",
    "deepaco_acs": "#f59e0b", "deepaco_acs_2opt": "#047857",
}
ORDER = list(LABELS)


def label(m):
    return LABELS.get(m, m)


def read_all(patterns):
    files = []
    for pat in patterns:
        files += sorted(glob.glob(pat)) or [pat]
    # ignora las tablas de resumen que genera este mismo script
    files = [f for f in dict.fromkeys(files) if not f.endswith("_resumen.csv")]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df.duplicated(["method", "seed", "instance", "T"]).sum()
    if dup:
        print(f"Aviso: {dup} filas repetidas (misma semilla/instancia/T); se conserva la primera.")
        df = df.drop_duplicates(["method", "seed", "instance", "T"])
    return df, files


def per_run(df):
    """Costo promedio sobre instancias, por método, semilla y T."""
    return df.groupby(["method", "seed", "T"], as_index=False)["cost"].mean()


def sorted_methods(methods):
    return sorted(methods, key=lambda m: ORDER.index(m) if m in ORDER else len(ORDER))


def summary(runs, T):
    sub = runs[runs["T"] == T]
    tab = sub.groupby("method")["cost"].agg(["count", "mean", "median", "std", "min", "max"])
    tab = tab.loc[sorted_methods(tab.index)]
    tab.index = [label(m) for m in tab.index]
    return tab.rename(columns={"count": "corridas", "mean": "media", "median": "mediana", "std": "desv_est"})


def paired(sub, a, b):
    pair = sub[[a, b]].dropna()
    if len(pair) < 2 or (pair[a] - pair[b]).abs().sum() == 0:
        return None
    stat, p = wilcoxon(pair[a], pair[b])
    diff = (pair[a] - pair[b]).mean()
    return (f"{label(b)} vs {label(a)} (Wilcoxon pareado, n={len(pair)}): W={stat:.1f}, p={p:.3g}, "
            f"diferencia media = {diff:+.4f} -> {label(b)} es {abs(100 * diff / pair[a].mean()):.2f}% "
            f"{'mejor' if diff > 0 else 'peor'} (rutas {'más cortas' if diff > 0 else 'más largas'})")


def compare(runs, T, baseline, reported):
    sub = runs[runs["T"] == T].pivot(index="seed", columns="method", values="cost")
    lines = []
    if {"aco", "deepaco"} <= set(sub.columns):
        lines.append(paired(sub, "aco", "deepaco"))
    if baseline in sub.columns:
        for m in sorted_methods(sub.columns):
            if m not in (baseline, "aco"):
                lines.append(paired(sub, baseline, m))
    for method, value in reported.items():
        if method in sub.columns:
            x = sub[method].dropna()
            stat, p = wilcoxon(x - value)
            lines.append(f"{label(method)} réplica vs reportado {value} (Wilcoxon 1 muestra, n={len(x)}): "
                         f"mediana={x.median():.4f}, W={stat:.1f}, p={p:.3g}")
    return [l for l in lines if l]


def plot(runs, n_ants, title, path):
    curve = runs.groupby(["method", "T"])["cost"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=150)
    for method in sorted_methods(curve["method"].unique()):
        g = curve[curve["method"] == method]
        x = g["T"] * n_ants
        c = COLORS.get(method)
        ax.plot(x, g["mean"], marker="o", ms=3, lw=1.6, color=c, label=label(method))
        ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Evaluaciones (hormigas × iteraciones)")
    ax.set_ylabel("Mejor longitud promedio")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="uno o varios CSV (se aceptan comodines como results/tsp100_as_*.csv)")
    p.add_argument("--T", type=int, default=None, help="iteración a comparar (por defecto la última)")
    p.add_argument("--n-ants", type=int, default=None, help="para el eje x de la curva")
    p.add_argument("--baseline", default="deepaco", help="método contra el que se comparan los demás")
    p.add_argument("--reported", nargs="*", default=[], help="valores del artículo, p.ej. aco=3.835 deepaco=3.809")
    p.add_argument("--name", default=None, help="nombre base de los archivos de salida en results/")
    args = p.parse_args()

    df, files = read_all(args.csv)
    size = int(df["size"].iloc[0])
    variants = "+".join(sorted(df["variant"].astype(str).unique()))
    n_ants = args.n_ants or {20: 20, 100: 20, 500: 50}.get(size, 1)
    T = args.T or int(df["T"].max())
    reported = {k: float(v) for k, v in (r.split("=") for r in args.reported)}
    runs = per_run(df)

    print(f"Archivos: {', '.join(os.path.basename(f) for f in files)}")
    print(f"TSP{size} | {variants} | T={T} | instancias por corrida: {df['instance'].nunique()}\n")
    print(summary(runs, T).round(4).to_string(), "\n")
    for line in compare(runs, T, args.baseline, reported):
        print(line)

    base = args.name or re.sub(r"_[ab]$", "", os.path.splitext(os.path.basename(files[0]))[0])
    out_dir = os.path.dirname(os.path.abspath(files[0]))
    out = os.path.join(out_dir, base)
    summary(runs, T).round(6).to_csv(out + "_resumen.csv")
    plot(runs, n_ants, f"TSP{size} — media ± d.e. de {runs['seed'].nunique()} corridas", out + "_curva.png")
    print(f"\nGuardado: {out}_resumen.csv y {out}_curva.png")


if __name__ == "__main__":
    main()
