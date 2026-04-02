import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
#from map_elements import GameMap
from optimization.map_elements import GameMap 

class Path:
    """
    Represents a path γ in 2D continuous space as defined in the thesis (Section 1.2).

    A path is stored as an ordered sequence of n waypoints (knots) p_i ∈ R²,
    where p_0 is the start point A and p_{n-1} is the goal point B.
    All optimization algorithms operate by modifying the inner waypoints
    p_1, ..., p_{n-2} while keeping the endpoints fixed.

    Attributes:
        points (np.ndarray): Array of shape (n_points, 2) containing [x, y]
                             coordinates of each waypoint.
    """
    def __init__(self, points: np.ndarray):
        # shape: (n_points, 2)
        self.points = points

    @property
    def n_points(self) -> int:
        """Returns the total number of waypoints including start and goal."""
        return self.points.shape[0]

    def copy(self) -> "Path":
        """Returns a deep copy of this path. Used by optimizers to avoid
        mutating candidate solutions during evaluation."""
        return Path(self.points.copy())

class PathPlanningProblem:
    """
    Encapsulates the formal path optimization problem from the thesis (Section 1.2).

    The problem is defined as minimizing the composite objective:
        F(γ) = C1 * L(γ) + C2 * E(γ) + C3 * R(γ)

    where:
        L(γ) — total Euclidean path length (eq. 1.2)
        E(γ) — cumulative terrain energy cost along the path (eq. 1.3)
        R(γ) — cumulative collision risk with respect to obstacles (eq. 1.4)
        C1, C2, C3 — scalar weights controlling trade-offs between objectives

    This class provides both the objective value and its gradient with respect
    to the inner waypoints, used by gradient-based optimizers.

    Attributes:
        map (GameMap):        The environment containing terrain and obstacles.
        start (np.ndarray):   Start point A = [x, y].
        goal (np.ndarray):    Goal point B = [x, y].
        c1, c2, c3 (float):   Objective weights for length, energy, and risk.
        risk_mode (str): Selects the collision risk formulation:
            "khatib"   — Khatib repulsion potential (default, eq. 1.5).
                         Provides non-zero gradient everywhere within d0.
            "original" — Original inverse-square formula (eq. 1.4).
                         Collapses near obstacles; kept for testing only.
    """
    def __init__(self,
                 game_map: "GameMap",
                 start: np.ndarray,
                 goal: np.ndarray,
                 c1: float,
                 c2: float,
                 c3: float,
                 risk_mode: str = "khatib"):
        self.map = game_map
        self.start = start
        self.goal = goal
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.risk_mode = risk_mode

    def path_length(self, path: Path) -> float:
        """
        Computes the total path length L(γ) as the sum of Euclidean distances
        between consecutive waypoints (thesis eq. 1.2):

            L(γ) = Σ ||p_i - p_{i+1}||₂  for i = 1..n-1

        Args:
            path: The path to evaluate.

        Returns:
            Scalar total length.
        """
        diffs = path.points[1:] - path.points[:-1]
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def path_energy(self, path: Path) -> float:
        """
        Computes arc-length-weighted terrain energy E(γ) using segment sampling.

        Instead of summing energy only at discrete waypoints, this method
        samples 10 points per segment and weights each segment's energy
        by its Euclidean length. This ensures:
        - A long segment through a swamp costs proportionally more than
            a short one, even if both have the same number of waypoints.
        - The gradient signal pushes waypoints toward lower-cost terrain
            rather than stalling in the interior of a high-cost zone.

        E(γ) = Σ_i ||p_{i+1} - p_i|| * mean(e(samples on segment i))

        Args:
            path: The path to evaluate.

        Returns:
            Scalar total arc-length-weighted energy.
        """
        total = 0.0
        samples_per_segment = 10

        for i in range(path.n_points - 1):
            p0 = path.points[i]
            p1 = path.points[i + 1]
            seg_len = float(np.linalg.norm(p1 - p0))

            seg_energy = 0.0
            for t in np.linspace(0, 1, samples_per_segment):
                p = (1.0 - t) * p0 + t * p1
                seg_energy += float(self.map.energy_at(p))

            total += seg_len * seg_energy / samples_per_segment

        return total

    def path_risk(self, path: Path) -> float:
        """
        Computes cumulative collision risk along the path using
        segment sampling (10 points per segment).

        Delegates to either the Khatib potential or the original
        inverse-square formula depending on self.risk_mode.
        """
        total = 0.0
        n = path.n_points
        samples_per_segment = 10

        # Select the correct point-wise risk function
        if self.risk_mode == "original":
            risk_fn = self.map.collision_risk_original_at
        else:
            risk_fn = self.map.collision_risk_at

        for i in range(n - 1):
            for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
                p = (1 - t) * path.points[i] + t * path.points[i + 1]
                total += risk_fn(p)

        total += risk_fn(path.points[-1])
        return total


    def objective(self, path: Path) -> float:
        """
        Evaluates the weighted composite objective F(γ) = C1*L + C2*E + C3*R.

        Components whose weight is zero are skipped entirely to avoid
        unnecessary computation — e.g. if c3=0, the risk field (which
        involves iterating over all obstacle distances for every segment
        sample) is never evaluated.
        """
        result = 0.0

        if self.c1 != 0:
            result += self.c1 * self.path_length(path)

        if self.c2 != 0:
            result += self.c2 * self.path_energy(path)

        if self.c3 != 0:
            result += self.c3 * self.path_risk(path)

        return result

    def gradient(self, path: Path) -> np.ndarray:
        """
        Computes ∇F with respect to inner waypoints.

        Gradient components for zero-weighted terms are skipped — their
        contribution to ∇F is identically zero, so computing them wastes
        time without affecting the optimizer's update step.
        """
        n = path.n_points
        grad = np.zeros_like(path.points)

        for i in range(1, n - 1):
            pi     = path.points[i]
            p_prev = path.points[i - 1]
            p_next = path.points[i + 1]

            g = np.zeros(2)

            if self.c1 != 0:
                v1 = pi - p_prev
                v2 = pi - p_next
                gL = (v1 / (np.linalg.norm(v1) + 1e-8) + v2 / (np.linalg.norm(v2) + 1e-8))
                g += self.c1 * gL

            if self.c2 != 0:
                gE = self._numerical_gradient_single_point(path, i, self.path_energy)
                g += self.c2 * gE

            if self.c3 != 0:
                gR = self.risk_gradient_at_waypoint(path, i)
                g += self.c3 * gR

            grad[i] = g

        return grad

    def risk_gradient_at_waypoint(self, path: Path, index: int) -> np.ndarray:
        """
        Gradient of the segment-sampled risk w.r.t. waypoint at index.

        Delegates gradient computation to either the Khatib or original
        formula depending on self.risk_mode.
        """
        samples = 10
        grad = np.zeros(2)
        pts = path.points
        n = path.n_points

        # Select the correct gradient function
        if self.risk_mode == "original":
            grad_fn = self.map.collision_risk_gradient_original_at
        else:
            grad_fn = self.map.collision_risk_gradient_at

        if index > 0:
            for t in np.linspace(0, 1, samples, endpoint=False):
                s = (1 - t) * pts[index - 1] + t * pts[index]
                grad += t * grad_fn(s)

        if index < n - 1:
            for t in np.linspace(0, 1, samples, endpoint=False):
                s = (1 - t) * pts[index] + t * pts[index + 1]
                grad += (1 - t) * grad_fn(s)

        return grad

    def _numerical_gradient_single_point(
        self,
        path: Path,
        index: int,
        func
    ) -> np.ndarray:
        """
        Estimates the gradient of a scalar function with respect to a single
        waypoint using central finite differences:

            ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)

        eps=0.5 is chosen to be large enough relative to the map scale (0–100)
        for accurate estimation, but small enough to remain local.

        Args:
            path:  Current path.
            index: Index of the waypoint to differentiate.
            func:  Scalar function Path → float to differentiate.

        Returns:
            np.ndarray of shape (2,) — [∂f/∂x, ∂f/∂y].
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
    Records the intermediate states of an optimization algorithm for visualization.

    After the algorithm finishes, the stored sequence of paths can be replayed
    to illustrate how the solution evolved over iterations — this corresponds
    to the workflow visualization described in the experiment methodology (Chapter 4).

    Attributes:
        paths (List[Path]):           Snapshots of the path at each recorded iteration.
        objective_values (List[float]): Corresponding objective function values F(γ).
    """
    def __init__(self):
        self.paths: List[Path] = []
        self.objective_values: List[float] = []

    def add(self, path: Path, obj: float):
        """
        Saves a copy of the current path and its objective value.

        A copy is saved (not a reference) so that subsequent modifications
        by the optimizer do not overwrite this snapshot.

        Args:
            path: Current path at this iteration.
            obj:  Value of F(γ) at this iteration.
        """
        self.paths.append(path.copy())
        self.objective_values.append(obj)

class PathOptimizer(ABC):
    """
    Abstract base class for all path optimization algorithms.

    Defines a common interface so that different algorithms
    (gradient-based, swarm-based) can be used interchangeably
    in the experiment runner and visualizer.

    Attributes:
        problem (PathPlanningProblem): The problem instance to solve.
        history (OptimizationHistory): Records intermediate paths for visualization.
    """
    def __init__(self, problem: PathPlanningProblem):
        self.problem = problem
        self.history = OptimizationHistory()

    @abstractmethod
    def optimize(self, initial_path: Path) -> Path:
        """
        Runs the optimization algorithm starting from initial_path.

        Args:
            initial_path: Starting path (typically a perturbed straight line).

        Returns:
            Best path found by the algorithm.
        """
        pass
