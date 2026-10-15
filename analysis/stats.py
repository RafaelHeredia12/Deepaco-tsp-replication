"""
Estadísticas de la réplica a partir de los CSV de experiments/run_tsp.py.

Para cada configuración (tamaño, variante, método) calcula, sobre las corridas
independientes, la media, mediana y desviación estándar del costo promedio
por corrida. Después compara:
  * ACO vs DeepACO con la prueba de rangos con signo de Wilcoxon (pareada por semilla).
  * Réplica vs valor reportado por los autores (Wilcoxon de una muestra), si se da --reported.
Además dibuja las curvas de evolución (costo vs número de evaluaciones), como la Fig. 2.

Ejemplo:
    python analysis/stats.py results/tsp20_as.csv --reported aco=3.835 deepaco=3.809
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

LABELS = {"aco": "ACO", "deepaco": "DeepACO"}
COLORS = {"aco": "#8a8a8a", "deepaco": "#c2410c"}


def per_run(df):
    """Costo promedio sobre instancias, por semilla, método y T."""
    return df.groupby(["method", "seed", "T"], as_index=False)["cost"].mean()


def summary(runs, T):
    sub = runs[runs["T"] == T]
    tab = sub.groupby("method")["cost"].agg(["count", "mean", "median", "std", "min", "max"])
    return tab.rename(columns={"count": "corridas"})


def compare(runs, T, reported):
    sub = runs[runs["T"] == T].pivot(index="seed", columns="method", values="cost")
    lines = []
    if {"aco", "deepaco"} <= set(sub.columns):
        paired = sub[["aco", "deepaco"]].dropna()
        stat, p = wilcoxon(paired["aco"], paired["deepaco"])
        diff = (paired["aco"] - paired["deepaco"]).mean()
        lines.append(f"ACO vs DeepACO (Wilcoxon pareado, n={len(paired)}): W={stat:.1f}, p={p:.3g}, "
                     f"mejora media de DeepACO = {diff:.4f} ({100 * diff / paired['aco'].mean():.2f}%)")
    for method, value in reported.items():
        if method in sub.columns:
            x = sub[method].dropna()
            stat, p = wilcoxon(x - value)
            lines.append(f"{LABELS.get(method, method)} réplica vs reportado {value} "
                         f"(Wilcoxon 1 muestra, n={len(x)}): mediana={x.median():.4f}, W={stat:.1f}, p={p:.3g}")
    return lines


def plot(runs, n_ants, title, path):
    curve = runs.groupby(["method", "T"])["cost"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=150)
    for method, g in curve.groupby("method"):
        x = g["T"] * n_ants
        ax.plot(x, g["mean"], marker="o", ms=3, lw=1.6, color=COLORS.get(method), label=LABELS.get(method, method))
        ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS.get(method), alpha=0.15, lw=0)
    ax.set_xlabel("Evaluaciones (hormigas × iteraciones)")
    ax.set_ylabel("Mejor longitud promedio")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--T", type=int, default=None, help="iteración a comparar (por defecto la última)")
    p.add_argument("--n-ants", type=int, default=None, help="para el eje x de la curva")
    p.add_argument("--reported", nargs="*", default=[], help="valores del artículo, p.ej. aco=3.835 deepaco=3.809")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    size, variant = df["size"].iloc[0], df["variant"].iloc[0]
    n_ants = args.n_ants or {20: 20, 100: 20, 500: 50}.get(int(size), 1)
    T = args.T or int(df["T"].max())
    reported = {k: float(v) for k, v in (r.split("=") for r in args.reported)}
    runs = per_run(df)

    print(f"TSP{size} | variante {variant} | T={T} | instancias por corrida: {df['instance'].nunique()}\n")
    print(summary(runs, T).round(4).to_string(), "\n")
    for line in compare(runs, T, reported):
        print(line)

    base = os.path.splitext(args.csv)[0]
    summary(runs, T).round(6).to_csv(base + "_resumen.csv")
    plot(runs, n_ants, f"TSP{size} — {variant.upper()} (media ± d.e. de {runs['seed'].nunique()} corridas)",
         base + "_curva.png")
    print(f"\nGuardado: {base}_resumen.csv y {base}_curva.png")


if __name__ == "__main__":
    main()
