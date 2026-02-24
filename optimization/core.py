import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
#from map_elements import GameMap
from optimization.map_elements import GameMap 

class Path:
    """
    Represents a path γ = [p1, ..., pn] with pi in R^2.
    """
    def __init__(self, points: np.ndarray):
        # shape: (n_points, 2)
        self.points = points

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def copy(self) -> "Path":
        return Path(self.points.copy())

class PathPlanningProblem:
    """
    Holds the formal problem: map, weights C1,C2,C3, start A, goal B.
    Provides L(γ), E(γ), R(γ), F(γ) and their gradients.
    """
    def __init__(self,
                 game_map: "GameMap",
                 start: np.ndarray,
                 goal: np.ndarray,
                 c1: float,
                 c2: float,
                 c3: float):
        self.map = game_map
        self.start = start
        self.goal = goal
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def path_length(self, path: Path) -> float:
        # L(γ)
        diffs = path.points[1:] - path.points[:-1]
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def path_energy(self, path: Path) -> float:
        # E(γ) = sum e(pi)
        energies = [self.map.energy_at(p) for p in path.points]
        return float(np.sum(energies))

    def path_risk(self, path: Path) -> float:
        # R(γ) = sum 1/(||pi - ti||^2 + eps)
        risks = [self.map.collision_risk_at(p) for p in path.points]
        return float(np.sum(risks))

    def objective(self, path: Path) -> float:
        # F(γ) = C1 L + C2 E + C3 R
        return (self.c1 * self.path_length(path) +
                self.c2 * self.path_energy(path) +
                self.c3 * self.path_risk(path))

    def gradient(self, path: Path) -> np.ndarray:
        """
        ∇F wrt inner waypoints pi (keep start, goal fixed).
        Returns array of shape (n_points, 2).
        Start and goal gradients are zero.
        """
        n = path.n_points
        grad = np.zeros_like(path.points)

        # Gradient of length term as in thesis (only inner points)
        for i in range(1, n - 1):
            pi = path.points[i]
            p_prev = path.points[i - 1]
            p_next = path.points[i + 1]

            v1 = pi - p_prev
            v2 = pi - p_next
            # avoid division by zero
            gL = v1 / (np.linalg.norm(v1) + 1e-8) + \
                 v2 / (np.linalg.norm(v2) + 1e-8)

            # Approximate energy and risk gradients by finite differences
            gE = self._numerical_gradient_single_point(
                path, i, self.path_energy
            )
            gR = self._numerical_gradient_single_point(
                path, i, self.path_risk
            )

            grad[i] = (self.c1 * gL +
                       self.c2 * gE +
                       self.c3 * gR)

        return grad

    def _numerical_gradient_single_point(
        self,
        path: Path,
        index: int,
        func
    ) -> np.ndarray:
        """
        Finite-difference gradient wrt single waypoint at given index.
        """
        eps = 1e-3
        base_path = path.copy()
        base_val = func(base_path)

        g = np.zeros(2)
        for dim in range(2):
            p_plus = base_path.copy()
            p_minus = base_path.copy()

            p_plus.points[index, dim] += eps
            p_minus.points[index, dim] -= eps

            f_plus = func(p_plus)
            f_minus = func(p_minus)

            g[dim] = (f_plus - f_minus) / (2 * eps)

        return g

class OptimizationHistory:
    """
    Stores intermediate paths for visualization of the workflow.
    """
    def __init__(self):
        self.paths: List[Path] = []
        self.objective_values: List[float] = []

    def add(self, path: Path, obj: float):
        self.paths.append(path.copy())
        self.objective_values.append(obj)

class PathOptimizer(ABC):
    """
    Abstract base for path optimization algorithms.
    """
    def __init__(self, problem: PathPlanningProblem):
        self.problem = problem
        self.history = OptimizationHistory()

    @abstractmethod
    def optimize(self, initial_path: Path) -> Path:
        pass
