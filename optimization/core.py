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

    '''
    def path_risk(self, path: Path) -> float:
        # R(γ) = sum 1/(||pi - ti||^2 + eps)
        risks = [self.map.collision_risk_at(p) for p in path.points]
        return float(np.sum(risks))
    '''
    def path_risk(self, path: Path) -> float:
        total = 0.0
        n = path.n_points
        samples_per_segment = 5   # interpolate between each pair of waypoints

        for i in range(n - 1):
            for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
                p = (1 - t) * path.points[i] + t * path.points[i + 1]
                total += self.map.collision_risk_at(p)

        total += self.map.collision_risk_at(path.points[-1])
        return total


    def objective(self, path: Path) -> float:
        # F(γ) = C1 L + C2 E + C3 R
        return (self.c1 * self.path_length(path) +
                self.c2 * self.path_energy(path) +
                self.c3 * self.path_risk(path))

    def gradient(self, path: Path) -> np.ndarray:
        n = path.n_points
        grad = np.zeros_like(path.points)

        for i in range(1, n - 1):
            pi = path.points[i]
            p_prev = path.points[i - 1]
            p_next = path.points[i + 1]

            # Analytical length gradient (thesis eq. 3.2)
            v1 = pi - p_prev
            v2 = pi - p_next
            gL = v1 / (np.linalg.norm(v1) + 1e-8) + \
                v2 / (np.linalg.norm(v2) + 1e-8)

            # Numerical energy gradient (terrain grid, finite diff is fine here)
            gE = self._numerical_gradient_single_point(path, i, self.path_energy)

            # Analytical risk gradient — NO finite differences
            #gR = self.map.collision_risk_gradient_at(pi)
            gR = self.risk_gradient_at_waypoint(path, i)

            grad[i] = self.c1 * gL + self.c2 * gE + self.c3 * gR

        return grad
    
    def risk_gradient_at_waypoint(self, path: Path, index: int) -> np.ndarray:
        """
        Gradient of segment-sampled risk w.r.t. waypoint at given index.
        Consistent with path_risk() which samples along segments.
        """
        samples = 10   # same resolution as path_risk
        grad = np.zeros(2)
        pts = path.points
        n = path.n_points

        # contribution from segment [p_{index-1}, p_{index}]
        if index > 0:
            for t in np.linspace(0, 1, samples, endpoint=False):
                s = (1 - t) * pts[index - 1] + t * pts[index]
                g = self.map.collision_risk_gradient_at(s)
                grad += t * g          # chain rule: ds/dp_i = t

        # contribution from segment [p_{index}, p_{index+1}]
        if index < n - 1:
            for t in np.linspace(0, 1, samples, endpoint=False):
                s = (1 - t) * pts[index] + t * pts[index + 1]
                g = self.map.collision_risk_gradient_at(s)
                grad += (1 - t) * g   # chain rule: ds/dp_i = (1-t)

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
        eps = 0.5
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
