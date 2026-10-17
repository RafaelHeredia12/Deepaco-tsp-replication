"""
Mejora propuesta: hibridación de DeepACO con búsqueda local 2-opt.

Corre las configuraciones de la mejora con exactamente el mismo protocolo que la réplica
(mismas instancias, semillas y puntos de control T), para compararlas de forma pareada
con los resultados de experiments/run_tsp.py.

Métodos:
  deepaco_2opt      DeepACO (Ant System) + 2-opt          -> mejora principal
  deepaco_acs_2opt  DeepACO con Ant Colony System + 2-opt  -> mejora + otra variante de ACO
  aco_2opt          ACO clásico (eta = 1/d) + 2-opt         -> control
  deepaco_acs       DeepACO con Ant Colony System, sin 2-opt (ablación, opcional)

Ejemplo:
    python experiments/run_mejora.py --size 100 --runs 30 --out results/mejora_tsp100.csv

Salida: CSV con las mismas columnas que la réplica
    size, variant, method, seed, instance, T, cost
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tsp  # noqa: E402  (también agrega deepaco/tsp al path)
from run_tsp import ACO, DEFAULTS, EPS, ROOT, gen_pyg_data, load_instances, load_model  # noqa: E402
from variants import make_acs, with_two_opt  # noqa: E402

ACS = make_acs(ACO)
METHODS = {
    # nombre: (clase, usa heurística aprendida)
    "deepaco_2opt":     (with_two_opt(ACO), True),
    "deepaco_acs_2opt": (with_two_opt(ACS), True),
    "aco_2opt":         (with_two_opt(ACO), False),
    "deepaco_acs":      (ACS, True),
}


@torch.no_grad()
def solve(coords, method, net, cfg, checkpoints, device):
    cls, learned = METHODS[method]
    pyg_data, distances = gen_pyg_data(coords, k_sparse=cfg["k_sparse"])
    common = dict(distances=distances, n_ants=cfg["n_ants"], decay=cfg["decay"],
                  alpha=cfg["alpha"], beta=cfg["beta"], device=device)
    if learned:
        heu = net.reshape(pyg_data, net(pyg_data)) + EPS
        aco = cls(heuristic=heu, **common)
    else:
        aco = cls(**common)
        aco.sparsify(cfg["k_sparse"])
    costs, done = [], 0
    for t in checkpoints:
        aco.run(t - done)
        done = t
        costs.append(float(aco.lowest_cost))
    return costs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, choices=sorted(DEFAULTS), required=True)
    p.add_argument("--methods", nargs="+", default=["deepaco_2opt", "deepaco_acs_2opt", "aco_2opt"],
                   choices=sorted(METHODS))
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--instances", type=int, default=100)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1, 10, 20, 30, 40, 50, 100])
    p.add_argument("--n-ants", type=int, default=None)
    p.add_argument("--k-sparse", type=int, default=None)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", default="cpu", help="2-opt corre en CPU; por defecto todo en CPU")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = dict(DEFAULTS[args.size])
    if args.n_ants: cfg["n_ants"] = args.n_ants
    if args.k_sparse: cfg["k_sparse"] = args.k_sparse
    cfg.update(decay=args.decay, alpha=args.alpha, beta=args.beta)
    checkpoints = sorted(set(args.checkpoints))

    out = args.out or os.path.join(ROOT, "results", f"mejora_tsp{args.size}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new_file = not os.path.exists(out)

    instances = load_instances(args.size, args.instances, args.device)
    net = load_model(args.size, args.device)

    print(f"MEJORA TSP{args.size} | {args.methods} | {cfg} | T={checkpoints} | {len(instances)} instancias | "
          f"semillas {args.first_seed}..{args.first_seed + args.runs - 1} | {args.device}")

    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["size", "variant", "method", "seed", "instance", "T", "cost"])
        for seed in range(args.first_seed, args.first_seed + args.runs):
            for method in args.methods:
                torch.manual_seed(seed)
                t0 = time.time()
                final = []
                for i, coords in enumerate(instances):
                    costs = solve(coords, method, net, cfg, checkpoints, args.device)
                    for t, c in zip(checkpoints, costs):
                        w.writerow([args.size, "mejora", method, seed, i, t, f"{c:.6f}"])
                    final.append(costs[-1])
                f.flush()
                print(f"  semilla {seed:2d} {method:16s} costo medio T={checkpoints[-1]}: "
                      f"{sum(final) / len(final):.4f}  ({time.time() - t0:.1f}s)")
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
