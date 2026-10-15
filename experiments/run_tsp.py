"""
Experimento principal de la réplica: ACO clásico vs DeepACO en TSP
(panel TSP de la Fig. 2 de Ye et al., 2023).

Cada "corrida" usa una semilla distinta y resuelve las mismas N instancias
de prueba. Para cada instancia se guarda el mejor costo encontrado en cada
punto de control de iteraciones (T). La métrica por corrida es el promedio
del mejor costo sobre las N instancias, igual que en el artículo.

Ejemplo:
    python experiments/run_tsp.py --size 20 --variant as --runs 30 --instances 100

Salida: results/tsp{size}_{variant}.csv con columnas
    size, variant, method, seed, instance, T, cost
"""
import argparse
import csv
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deepaco", "tsp"))

from aco import ACO          # noqa: E402  (código original de los autores)
from net import Net          # noqa: E402
from utils import gen_pyg_data  # noqa: E402

EPS = 1e-10  # mismo valor que en test.ipynb de los autores

# Configuración por tamaño tomada del código original (train.ipynb / test.ipynb)
# y del Apéndice D.1 del artículo (k vecinos más cercanos).
DEFAULTS = {
    20:  dict(k_sparse=10, n_ants=20),
    100: dict(k_sparse=20, n_ants=20),
    500: dict(k_sparse=50, n_ants=50),
}

VARIANTS = {
    "as":      dict(elitist=False, min_max=False),  # Ant System
    "elitist": dict(elitist=True,  min_max=False),  # Elitist Ant System
    "mmas":    dict(elitist=False, min_max=True),   # MAX-MIN Ant System
}


def load_instances(size, n_instances, device):
    path = os.path.join(ROOT, "deepaco", "data", "tsp", f"testDataset-{size}.pt")
    coords = torch.load(path)
    return coords[:n_instances].to(device)


def load_model(size, device):
    net = Net().to(device)
    path = os.path.join(ROOT, "deepaco", "pretrained", "tsp", f"tsp{size}.pt")
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


@torch.no_grad()
def solve(coords, method, net, cfg, variant, checkpoints, device):
    """Corre una colonia sobre una instancia y devuelve el mejor costo en cada T."""
    pyg_data, distances = gen_pyg_data(coords, k_sparse=cfg["k_sparse"])
    common = dict(distances=distances, n_ants=cfg["n_ants"], decay=cfg["decay"],
                  alpha=cfg["alpha"], beta=cfg["beta"], device=device, **VARIANTS[variant])
    if method == "deepaco":
        heu = net.reshape(pyg_data, net(pyg_data)) + EPS
        aco = ACO(heuristic=heu, **common)
    else:
        aco = ACO(**common)
        aco.sparsify(cfg["k_sparse"])  # misma esparsificación que DeepACO (Apéndice D.1)

    costs, done = [], 0
    for t in checkpoints:
        aco.run(t - done)
        done = t
        costs.append(float(aco.lowest_cost))
    return costs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, choices=sorted(DEFAULTS), required=True)
    p.add_argument("--variant", choices=sorted(VARIANTS), default="as")
    p.add_argument("--methods", nargs="+", default=["aco", "deepaco"], choices=["aco", "deepaco"])
    p.add_argument("--runs", type=int, default=30, help="corridas independientes (semillas 0..runs-1)")
    p.add_argument("--first-seed", type=int, default=0, help="para repartir corridas entre sesiones")
    p.add_argument("--instances", type=int, default=100, help="instancias de prueba (el artículo usa 100)")
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1, 10, 20, 30, 40, 50, 100],
                   help="iteraciones T donde se registra el mejor costo")
    p.add_argument("--n-ants", type=int, default=None)
    p.add_argument("--k-sparse", type=int, default=None)
    p.add_argument("--decay", type=float, default=0.9, help="factor de retención de feromona (1 - evaporación)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = dict(DEFAULTS[args.size])
    if args.n_ants: cfg["n_ants"] = args.n_ants
    if args.k_sparse: cfg["k_sparse"] = args.k_sparse
    cfg.update(decay=args.decay, alpha=args.alpha, beta=args.beta)
    checkpoints = sorted(set(args.checkpoints))

    out = args.out or os.path.join(ROOT, "results", f"tsp{args.size}_{args.variant}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new_file = not os.path.exists(out)

    instances = load_instances(args.size, args.instances, args.device)
    net = load_model(args.size, args.device) if "deepaco" in args.methods else None

    print(f"TSP{args.size} | {args.variant} | {cfg} | T={checkpoints} | "
          f"{len(instances)} instancias | semillas {args.first_seed}..{args.first_seed + args.runs - 1} | {args.device}")

    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["size", "variant", "method", "seed", "instance", "T", "cost"])
        for seed in range(args.first_seed, args.first_seed + args.runs):
            for method in args.methods:
                torch.manual_seed(seed)  # misma semilla para ambos métodos -> comparación pareada
                t0 = time.time()
                final = []
                for i, coords in enumerate(instances):
                    costs = solve(coords, method, net, cfg, args.variant, checkpoints, args.device)
                    for t, c in zip(checkpoints, costs):
                        w.writerow([args.size, args.variant, method, seed, i, t, f"{c:.6f}"])
                    final.append(costs[-1])
                f.flush()
                print(f"  semilla {seed:2d} {method:8s} costo medio T={checkpoints[-1]}: "
                      f"{sum(final) / len(final):.4f}  ({time.time() - t0:.1f}s)")
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
