"""
Figuras y tabla resumen para el póster, a partir de los CSV en results/.

Genera solo lo que tenga datos disponibles:
  results/figuras/fig1_tsp20_vs_articulo.png   Réplica vs valores publicados (TSP20, Ant System)
  results/figuras/fig2_mejora_deepaco.png      Mejora de DeepACO sobre ACO por variante y tamaño
  results/figuras/fig3_mejora_2opt.png         Réplica vs mejora con 2-opt (TSP100)
  results/figuras/fig4_curvas_2opt.png         Evolución de las variantes con 2-opt (TSP100)
  results/figuras/tabla_resumen.csv            Media, mediana y d.e. de todas las configuraciones

Uso:
    python analysis/figuras_poster.py            (busca los CSV en results/)
    python analysis/figuras_poster.py --dir otra/carpeta
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paleta categórica validada (claro); cada método conserva su color en todas las figuras.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {
    "deepaco": "#2a78d6",          # azul
    "aco": "#eb6834",              # naranja
    "aco_2opt": "#1baf7a",         # aqua
    "deepaco_2opt": "#6250d6",     # violeta
    "deepaco_acs_2opt": "#e87ba4", # magenta
    "deepaco_acs": "#eda100",      # amarillo
}
LABEL = {
    "aco": "ACO clásico",
    "deepaco": "DeepACO (réplica)",
    "aco_2opt": "ACO clásico + 2-opt",
    "deepaco_2opt": "DeepACO + 2-opt",
    "deepaco_acs_2opt": "DeepACO + ACS + 2-opt",
    "deepaco_acs": "DeepACO + ACS",
}
VARIANT = {"as": "Ant System", "elitist": "Elitist AS", "mmas": "MAX-MIN AS"}
N_ANTS = {20: 20, 100: 20, 500: 50}
# Valores publicados por los autores (deepaco/tsp/test.ipynb: TSP20, Ant System, 20 hormigas)
PAPER_TSP20 = {
    "T": [1, 10, 20, 30, 40, 50, 100],
    "aco": [5.255, 4.207, 4.042, 3.949, 3.911, 3.889, 3.835],
    "deepaco": [3.932, 3.823, 3.817, 3.812, 3.811, 3.811, 3.809],
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(directory):
    frames = []
    for f in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        if f.endswith("_resumen.csv") or os.path.basename(f).startswith("tabla_"):
            continue
        d = pd.read_csv(f)
        if {"size", "variant", "method", "seed", "instance", "T", "cost"} <= set(d.columns):
            frames.append(d)
    if not frames:
        raise SystemExit(f"No hay CSV de resultados en {directory}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["size", "variant", "method", "seed", "instance", "T"])


def per_run(df):
    return df.groupby(["size", "variant", "method", "seed", "T"], as_index=False)["cost"].mean()


def style_axes(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_tsp20(runs, out):
    sub = runs[(runs["size"] == 20) & (runs["variant"] == "as")]
    if sub.empty:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style_axes(ax)
    for m in ["aco", "deepaco"]:
        g = sub[sub["method"] == m].groupby("T")["cost"].agg(["mean", "std"]).reset_index()
        x = g["T"] * N_ANTS[20]
        ax.plot(x, g["mean"], color=COLOR[m], lw=2, marker="o", ms=4, label=f"{LABEL[m]} — 30 corridas")
        ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=COLOR[m], alpha=0.18, lw=0)
        px = [t * N_ANTS[20] for t in PAPER_TSP20["T"]]
        ax.scatter(px, PAPER_TSP20[m], s=46, facecolors="none", edgecolors=INK, lw=1.3, zorder=5,
                   label="Valores del artículo" if m == "aco" else None)
    ax.set_xlabel("Evaluaciones (hormigas × iteraciones)")
    ax.set_ylabel("Mejor longitud promedio del tour")
    ax.set_title("TSP20, Ant System: réplica vs. artículo", loc="left")
    ax.set_xlim(-30, 2100)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_variantes(runs, out):
    base = runs[runs["method"].isin(["aco", "deepaco"]) & runs["variant"].isin(VARIANT)]
    if base.empty:
        return False
    last = base[base["T"] == base.groupby(["size", "variant"])["T"].transform("max")]
    piv = last.pivot_table(index=["size", "variant", "seed"], columns="method", values="cost").dropna()
    if piv.empty:
        return False
    piv["mejora"] = 100 * (piv["aco"] - piv["deepaco"]) / piv["aco"]
    stats = piv.groupby(["size", "variant"])["mejora"].agg(["mean", "std"]).reset_index()
    sizes = sorted(stats["size"].unique())
    variants = [v for v in VARIANT if v in set(stats["variant"])]
    size_color = {20: "#86b6ef", 100: "#2a78d6", 500: "#104281"}  # rampa azul ordinal: más oscuro = más ciudades
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style_axes(ax)
    width = 0.8 / len(sizes)
    labeled = set()
    for i, s in enumerate(sizes):
        for j, v in enumerate(variants):
            row = stats[(stats["size"] == s) & (stats["variant"] == v)]
            if row.empty:
                continue
            x = j + (i - (len(sizes) - 1) / 2) * width
            val = float(row["mean"].iloc[0])
            ax.bar(x, val, width * 0.92, color=size_color.get(s, INK2),
                   label=None if s in labeled else f"TSP{s}")
            labeled.add(s)
            ax.annotate(f"{val:.1f}%", (x, val), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(variants)), [VARIANT[v] for v in variants])
    ax.set_ylabel("Rutas más cortas con DeepACO vs. ACO (%)")
    ax.set_title("Mejora de DeepACO sobre ACO clásico (mismas hormigas e iteraciones)", loc="left", fontsize=11)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.legend(frameon=False, title=None)
    if 500 in sizes:
        fig.text(0.01, -0.06, "TSP20 y TSP100: 100 instancias, T = 100. TSP500: solo Ant System, 20 instancias,\n"
                 "T = 50 (protocolo reducido por costo). 30 corridas en todos los casos.", fontsize=8, color=INK2)
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_mejora(runs, out, size=100):
    methods = ["aco", "deepaco", "aco_2opt", "deepaco_2opt", "deepaco_acs_2opt"]
    sub = runs[(runs["size"] == size) & runs["method"].isin(methods)
               & runs["variant"].isin(["as", "mejora"])]
    if not {"deepaco", "deepaco_2opt"} <= set(sub["method"]):
        return False
    T = sub["T"].max()
    last = sub[sub["T"] == T].groupby("method")["cost"].agg(["mean", "std"]).reindex(methods).dropna()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    y = range(len(last))[::-1]
    for yi, (m, row) in zip(y, last.iterrows()):
        ax.errorbar(row["mean"], yi, xerr=row["std"], fmt="o", ms=9, color=COLOR[m], ecolor=COLOR[m],
                    elinewidth=2, capsize=0)
        ax.annotate(f"{row['mean']:.3f}", (row["mean"], yi), xytext=(10, -4), textcoords="offset points",
                    fontsize=10, color=INK)
    ax.set_yticks(list(y), [LABEL[m] for m in last.index])
    ax.set_xlabel(f"Longitud promedio del tour en T = {T} (media ± d.e., 30 corridas)")
    ax.set_title(f"TSP{size}: efecto de la búsqueda local 2-opt", loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_curvas_2opt(runs, out, size=100):
    """Evolución de las variantes con 2-opt: muestra en qué momento ACO + 2-opt alcanza a DeepACO."""
    methods = ["deepaco_2opt", "deepaco_acs_2opt", "aco_2opt"]
    sub = runs[(runs["size"] == size) & runs["method"].isin(methods)]
    if not set(methods) <= set(sub["method"]):
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style_axes(ax)
    for m in methods:
        g = sub[sub["method"] == m].groupby("T")["cost"].agg(["mean", "std"]).reset_index()
        x = g["T"] * N_ANTS[size]
        ax.plot(x, g["mean"], color=COLOR[m], lw=2, marker="o", ms=4, label=LABEL[m])
        ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=COLOR[m], alpha=0.18, lw=0)
    ax.set_xlabel("Evaluaciones (hormigas × iteraciones)")
    ax.set_ylabel("Mejor longitud promedio del tour")
    ax.set_title(f"TSP{size} con 2-opt: DeepACO arranca mejor, ACO lo alcanza", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return True


def tabla(runs, out):
    last = runs[runs["T"] == runs.groupby(["size", "variant"])["T"].transform("max")]
    tab = (last.groupby(["size", "variant", "method"])
           .agg(T=("T", "max"), corridas=("cost", "count"), media=("cost", "mean"),
                mediana=("cost", "median"), desv_est=("cost", "std"))
           .reset_index())
    tab.insert(2, "metodo", tab.pop("method").map(lambda m: LABEL.get(m, m)))
    tab.round(4).to_csv(out, index=False)
    return tab


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.path.join(ROOT, "results"))
    args = p.parse_args()
    out_dir = os.path.join(args.dir, "figuras")
    os.makedirs(out_dir, exist_ok=True)
    runs = per_run(load(args.dir))

    made = []
    if fig_tsp20(runs, os.path.join(out_dir, "fig1_tsp20_vs_articulo.png")):
        made.append("fig1_tsp20_vs_articulo.png")
    if fig_variantes(runs, os.path.join(out_dir, "fig2_mejora_deepaco.png")):
        made.append("fig2_mejora_deepaco.png")
    if fig_mejora(runs, os.path.join(out_dir, "fig3_mejora_2opt.png")):
        made.append("fig3_mejora_2opt.png")
    if fig_curvas_2opt(runs, os.path.join(out_dir, "fig4_curvas_2opt.png")):
        made.append("fig4_curvas_2opt.png")
    tab = tabla(runs, os.path.join(out_dir, "tabla_resumen.csv"))
    made.append("tabla_resumen.csv")
    print(tab.round(4).to_string(index=False))
    print("\nGenerado en", out_dir + ":", ", ".join(made))


if __name__ == "__main__":
    main()
