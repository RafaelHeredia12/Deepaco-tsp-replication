"""
Variantes usadas en la mejora propuesta (trabajo del equipo, no de los autores).

- ACS: Ant Colony System (Dorigo & Gambardella, 1997) implementado sobre la clase ACO
  original de los autores.
- with_two_opt: hibrida cualquier colonia con búsqueda local 2-opt. Cada ruta construida
  por una hormiga se mejora con 2-opt ANTES de calcular su costo y depositar feromona.
  El 2-opt es la implementación de los autores (deepaco/tsp_nls/two_opt.py, licencia MIT).
"""
import importlib.util
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Se carga two_opt.py por ruta para no mezclar los módulos net/utils de tsp/ y tsp_nls/
_spec = importlib.util.spec_from_file_location(
    "deepaco_two_opt", os.path.join(ROOT, "deepaco", "tsp_nls", "two_opt.py"))
_two_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_two_opt)
batched_two_opt = _two_opt.batched_two_opt_python


def make_acs(ACO):
    """Construye la clase ACS a partir de la clase ACO original."""

    class ACS(ACO):
        """Ant Colony System.

        - Regla pseudoaleatoria proporcional: con probabilidad q0 la hormiga toma el arco
          con mayor tau^alpha * eta^beta; si no, muestrea como en Ant System.
        - Actualización local: tau <- (1 - xi) tau + xi tau0 en los arcos recorridos.
        - Actualización global: solo la mejor ruta encontrada hasta el momento deposita
          feromona: tau <- (1 - rho) tau + rho / L_best.
        - tau0 = 1 / (n * L), con L la longitud media de las rutas de la primera iteración.
        """

        def __init__(self, distances, q0=0.9, rho=0.1, xi=0.1, **kw):
            kw.pop("elitist", None); kw.pop("min_max", None)
            super().__init__(distances, **kw)
            self.q0, self.rho, self.xi = q0, rho, xi
            self.tau0 = None

        def pick_move(self, prev, mask, require_prob):
            w = (self.pheromone[prev] ** self.alpha) * (self.heuristic[prev] ** self.beta) * mask
            greedy = w.argmax(dim=1)
            sampled = torch.distributions.Categorical(w, validate_args=False).sample()
            use_greedy = torch.rand(len(prev), device=prev.device) < self.q0
            return torch.where(use_greedy, greedy, sampled), None

        @torch.no_grad()
        def run(self, n_iterations):
            for _ in range(n_iterations):
                paths = self.gen_path(require_prob=False)
                costs = self.gen_path_costs(paths)
                best_cost, best_idx = costs.min(dim=0)
                if best_cost < self.lowest_cost:
                    self.lowest_cost = best_cost
                    self.shortest_path = paths[:, best_idx].clone()
                if self.tau0 is None:
                    self.tau0 = 1.0 / (self.problem_size * float(costs.mean()))
                    self.pheromone.fill_(self.tau0)
                u = paths.flatten()
                v = torch.roll(paths, 1, 0).flatten()
                self.pheromone[u, v] = (1 - self.xi) * self.pheromone[u, v] + self.xi * self.tau0
                self.pheromone[v, u] = self.pheromone[u, v]
                b = self.shortest_path
                bv = torch.roll(b, 1)
                self.pheromone[b, bv] = (1 - self.rho) * self.pheromone[b, bv] + self.rho / float(self.lowest_cost)
                self.pheromone[bv, b] = self.pheromone[b, bv]
            return self.lowest_cost

    return ACS


def with_two_opt(cls, max_iterations=1000):
    """Devuelve una subclase de `cls` cuyas hormigas aplican 2-opt a su ruta."""

    class TwoOpt(cls):
        def gen_path(self, require_prob=False):
            paths = super().gen_path(require_prob)
            dist = self.distances.detach().cpu().numpy()
            tours = batched_two_opt(dist, paths.T.cpu().numpy(), max_iterations=max_iterations)
            return torch.from_numpy(tours.astype(np.int64)).T.contiguous().to(paths.device)

    TwoOpt.__name__ = cls.__name__ + "_2opt"
    return TwoOpt
