import numpy as np
from optimization.core import PathOptimizer, PathPlanningProblem, Path
from abc import ABC, abstractmethod
from typing import List, Tuple

class GradientDescentOptimizer(PathOptimizer):
    """
    Simple gradient descent on inner waypoints.
    """
    def __init__(self,
                 problem: PathPlanningProblem,
                 step_size: float = 0.01,
                 max_iters: int = 500,
                 tolerance: float = 1e-4):
        super().__init__(problem)
        self.step_size = step_size
        self.max_iters = max_iters
        self.tolerance = tolerance

    def optimize(self, initial_path: Path) -> Path:
        path = initial_path.copy()

        for it in range(self.max_iters):
            obj = self.problem.objective(path)
            self.history.add(path, obj)

            grad = self.problem.gradient(path)
            grad_norm = np.linalg.norm(grad[1:-1])  # ignore endpoints

            if grad_norm < self.tolerance:
                break

            # gradient step on inner points
            new_points = path.points.copy()
            new_points[1:-1] -= self.step_size * grad[1:-1]
            path.points = new_points

        return path
    
class AdamPathOptimizer(PathOptimizer):
    """
    Gradient-based path optimizer using the Adam update rule.

    Adam is selected over plain gradient descent
    for its adaptive per-parameter learning rates and momentum, which help
    navigate the non-convex objective function described in Section 1.3 of
    the thesis. It combines:
      - First moment (momentum): smooths gradient direction
      - Second moment (RMSprop): scales step size per parameter

    The update rule at iteration t (thesis Section 3.2.3):
        m_t = β1 * m_{t-1} + (1 - β1) * ∇F
        v_t = β2 * v_{t-1} + (1 - β2) * ∇F²
        m̂ = m_t / (1 - β1^t)          ← bias correction
        v̂ = v_t / (1 - β2^t)
        p ← p - lr * m̂ / (√v̂ + ε)

    Only inner waypoints are updated; start and goal are kept fixed.

    Args:
        problem:    The path planning problem instance.
        lr:         Learning rate (step size). Must be on the scale of the map.
        beta1:      Exponential decay for first moment (default 0.9).
        beta2:      Exponential decay for second moment (default 0.999).
        eps:        Numerical stability constant (default 1e-8).
        max_iters:  Maximum number of gradient steps.
        tolerance:  Early stopping threshold on the inner gradient norm.
    """
    def __init__(self,
                 problem: PathPlanningProblem,
                 lr: float = 1.0,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 max_iters: int = 500,
                 tolerance: float = 1e-6):
        super().__init__(problem)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iters = max_iters
        self.tolerance = tolerance

    def optimize(self, initial_path: Path) -> Path:
        """
        Runs Adam optimization starting from initial_path.

        The initial path must NOT be a perfect straight line because the
        analytical length gradient (thesis eq. 3.2) cancels to zero for
        collinear waypoints. A small perturbation must be applied before
        calling this method (see PathPlanningExperiment.run_gradient).

        Args:
            initial_path: Starting path with perturbed inner waypoints.

        Returns:
            Locally optimized path minimizing F(γ) = C1*L + C2*E + C3*R.
        """
        path = initial_path.copy()

        m = np.zeros_like(path.points)
        v = np.zeros_like(path.points)

        t = 0
        for _ in range(self.max_iters):
            t += 1
            obj = self.problem.objective(path)
            self.history.add(path, obj)

            grad = self.problem.gradient(path)

            # stop if gradient on inner points is small
            grad_inner = grad[1:-1]
            if np.linalg.norm(grad_inner) < self.tolerance:
                print(f"Adam converged at iteration {_}, obj={obj:.4f}")
                break

            # Adam update only on inner waypoints (keep endpoints fixed)
            m[1:-1] = self.beta1 * m[1:-1] + (1.0 - self.beta1) * grad_inner
            v[1:-1] = self.beta2 * v[1:-1] + (1.0 - self.beta2) * (grad_inner ** 2)

            # bias correction
            m_hat = m[1:-1] / (1.0 - self.beta1 ** t)
            v_hat = v[1:-1] / (1.0 - self.beta2 ** t)

            step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            new_points = path.points.copy()
            new_points[1:-1] -= step  # gradient descent direction
            
            # keep waypoints within map bounds
            new_points[1:-1, 0] = np.clip(
                new_points[1:-1, 0], 0, self.problem.map.width
            )
            new_points[1:-1, 1] = np.clip(
                new_points[1:-1, 1], 0, self.problem.map.height
            )
            path.points = new_points

        return path

class PSOParticle:
    """
    Represents a single particle in the Particle Swarm Optimization algorithm.

    In the context of path planning, each particle's position is a candidate
    Path rather than a point in R^n. The particle moves through the space of
    all possible paths between the fixed start and goal points.

    Attributes:
        position (Path):        Current candidate path (the particle's location).
        velocity (np.ndarray):  Per-waypoint velocity, shape (n_points, 2).
                                Drives how waypoints move between iterations.
        best_position (Path):   Best path this particle has personally visited.
        best_value (float):     Objective value at best_position.
    """
    def __init__(self, path: Path):
        self.position = path
        self.velocity = np.zeros_like(path.points)
        self.best_position = path.copy()
        self.best_value = float("inf")

class ParticleSwarmOptimizer(PathOptimizer):
    """
    Population-based path optimizer using Particle Swarm Optimization (PSO).

    Implements the algorithm described in thesis Section 3.3. Each particle
    represents a candidate path and updates its trajectory through path-space
    based on:
      - Its own inertia (previous velocity)
      - Cognitive influence: attraction toward its personal best path
      - Social influence: attraction toward the global best path found by any particle

    Velocity update rule (thesis eq. 3.14):
        v^{t+1} = ω * v^t
                + φ_p * R_p * (p_best - x^t)   ← cognitive
                + φ_g * R_g * (g_best - x^t)   ← social

    Position update (thesis eq. 3.15):
        x^{t+1} = x^t + v^{t+1}

    Args:
        problem:     The path planning problem instance.
        n_particles: Number of particles (candidate paths) in the swarm.
        max_iters:   Number of swarm iterations.
        omega:       Inertia weight — controls how much previous velocity persists.
        phi_p:       Cognitive weight — attraction to particle's own best.
        phi_g:       Social weight — attraction to swarm's global best.
    """
    def __init__(self,
                 problem: PathPlanningProblem,
                 n_particles: int = 50,
                 max_iters: int = 500,
                 omega: float = 0.4,
                 phi_p: float = 1.0,
                 phi_g: float = 1.0):
        super().__init__(problem)
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g

        self.particles: List[PSOParticle] = []
        self.global_best_path: Path | None = None
        self.global_best_value: float = float("inf")

    def _random_path_like(self, template: Path) -> Path:
        """
        Creates a randomized candidate path with the same start, goal,
        and number of waypoints as the template.

        Inner waypoints are initialized as a perturbed straight line
        (not fully random) to keep particles in a meaningful search region.
        Fully random initialization causes most particles to start with
        extremely long paths that push the swarm toward the straight-line
        solution dominated by the length term.

        Args:
            template: Reference path providing structure (n_points, start, goal).

        Returns:
            A new randomized Path between the same start and goal.
        """
        n = template.n_points
        pts = np.zeros_like(template.points)
        pts[0] = template.points[0]
        pts[-1] = template.points[-1]

        for i in range(1, n - 1):
            t = i / (n - 1)
            base = (1 - t) * template.points[0] + t * template.points[-1]
            noise = np.random.uniform(-20, 20, size=2)  # controlled perturbation
            pts[i] = np.clip(
                base + noise,
                [0, 0],
                [self.problem.map.width, self.problem.map.height]
                )   
        return Path(pts)

    def _init_swarm(self, template: Path):
        """
        Initializes all particles with random paths and evaluates their
        starting objective values. Identifies the initial global best.

        Args:
            template: Reference path for structure (n_points, start, goal).
        """
        self.particles = []
        self.global_best_path = None
        self.global_best_value = float("inf")

        for _ in range(self.n_particles):
            p = self._random_path_like(template)
            particle = PSOParticle(p)
            val = self.problem.objective(p)
            particle.best_value = val
            self.particles.append(particle)

            if val < self.global_best_value:
                self.global_best_value = val
                self.global_best_path = p.copy()

    def _reorder_waypoints(self, path: Path):
        """
        Re-sorts inner waypoints by their projection onto the start→goal
        axis. This prevents waypoints from crossing each other and forming
        loops while still allowing them to deviate sideways freely.
        """
        pts = path.points
        start = pts[0]
        goal  = pts[-1]

        direction = goal - start
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Project each inner waypoint onto the start→goal axis
        projections = [
            np.dot(pts[i] - start, direction)
            for i in range(1, path.n_points - 1)
        ]

        # Sort inner waypoints by their projection value
        inner_sorted = sorted(
            zip(projections, pts[1:-1]),
            key=lambda x: x[0]
        )

        for i, (_, p) in enumerate(inner_sorted):
            path.points[i + 1] = p


    def optimize(self, initial_path: Path) -> Path:
        """
        Runs PSO to find an optimized path.

        At each iteration every particle updates its velocity toward the
        personal and global best positions, then moves accordingly.
        The global best is updated whenever any particle finds a lower
        objective value.

        Waypoints are clamped to map bounds after each position update
        to maintain feasibility of candidate solutions.

        Args:
            initial_path: Used as a structural template for particle initialization.

        Returns:
            Best path found across all particles and all iterations.
        """
        self._init_swarm(initial_path)

        for it in range(self.max_iters):
            # store global best into history for visualization
            if self.global_best_path is not None:
                self.history.add(self.global_best_path, self.global_best_value)

                # extra safety: if somehow still None, break
            if self.global_best_path is None:
                break
            g_best = self.global_best_path.points     # local non‑optional

            for particle in self.particles:
                x = particle.position.points
                v = particle.velocity
                p_best = particle.best_position.points

                r_p = np.random.rand(*x.shape)
                r_g = np.random.rand(*x.shape)

                v = (self.omega * v +
                     self.phi_p * r_p * (p_best - x) +
                     self.phi_g * r_g * (g_best - x))

                # endpoints stay fixed
                v[0] = 0.0
                v[-1] = 0.0

                x_new = x + v
                # clamp to map bounds
                x_new[:, 0] = np.clip(x_new[:, 0], 0, self.problem.map.width)
                x_new[:, 1] = np.clip(x_new[:, 1], 0, self.problem.map.height)

                particle.velocity = v
                particle.position.points = x_new

                self._reorder_waypoints(particle.position)

                val = self.problem.objective(particle.position)

                if val < particle.best_value:
                    particle.best_value = val
                    particle.best_position = particle.position.copy()

                if val < self.global_best_value:
                    self.global_best_value = val
                    self.global_best_path = particle.position.copy()

        if self.global_best_path is None:
        # fallback: return initial_path or raise error
            return initial_path.copy()
        
        return self.global_best_path.copy()