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
    Gradient-based optimizer using Adam updates on inner waypoints.
    """
    def __init__(self,
                 problem: PathPlanningProblem,
                 lr: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 max_iters: int = 500,
                 tolerance: float = 1e-4):
        super().__init__(problem)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iters = max_iters
        self.tolerance = tolerance

    def optimize(self, initial_path: Path) -> Path:
        path = initial_path.copy()

        # moments (same shape as points)
        m = np.zeros_like(path.points)   # first moment
        v = np.zeros_like(path.points)   # second raw moment

        t = 0
        for _ in range(self.max_iters):
            t += 1
            obj = self.problem.objective(path)
            self.history.add(path, obj)

            grad = self.problem.gradient(path)

            # stop if gradient on inner points is small
            grad_inner = grad[1:-1]
            if np.linalg.norm(grad_inner) < self.tolerance:
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

            path.points = new_points

        return path

class PSOParticle:
    """
    One particle for PSO: position is a full Path.
    """
    def __init__(self, path: Path):
        self.position = path
        self.velocity = np.zeros_like(path.points)
        self.best_position = path.copy()
        self.best_value = float("inf")

class ParticleSwarmOptimizer(PathOptimizer):
    """
    Basic single-objective PSO on waypoint positions.
    """
    def __init__(self,
                 problem: PathPlanningProblem,
                 n_particles: int = 20,
                 max_iters: int = 300,
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
        Random path with same start/goal and same number of points.
        Middle points random in map bounds.
        """
        n = template.n_points
        pts = np.zeros_like(template.points)
        pts[0] = template.points[0]
        pts[-1] = template.points[-1]

        for i in range(1, n - 1):
            pts[i, 0] = np.random.uniform(0, self.problem.map.width)
            pts[i, 1] = np.random.uniform(0, self.problem.map.height)

        return Path(pts)

    def _init_swarm(self, template: Path):
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

    def optimize(self, initial_path: Path) -> Path:
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