import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

class GameMap:
    """
    Continuous 2D map with terrain and obstacles.
    Provides energy_at(x) and collision_risk_at(x).
    """
    def __init__(self,
                 width: float,
                 height: float,
                 terrain: "TerrainField",
                 obstacles: List["Obstacle"],
                 risk_epsilon: float = 0.000001):
        self.width = width
        self.height = height
        self.terrain = terrain
        self.obstacles = obstacles
        self.risk_epsilon = risk_epsilon

    def energy_at(self, point: np.ndarray) -> float:
        """
        Energy cost e(x) based on terrain type.
        """
        return self.terrain.energy_at(point)

    def nearest_obstacle_distance(self, point: np.ndarray) -> float:
        """
        Distance to nearest obstacle surface (>=0 outside, 0 inside).
        """
        dists = [obs.distance_to(point) for obs in self.obstacles]
        if not dists:
            return float("inf")
        return min(dists)

    def collision_risk_at(self, point: np.ndarray) -> float:
        """
        R term: 1 / (||p - t||^2 + eps).
        Here t is nearest obstacle surface point, approximated via distance.
        """
        d = self.nearest_obstacle_distance(point)
        d_safe = max(d, 1e-3)       # avoid /0 but keep gradient
        influence_radius = 20.0     # how far obstacles "push" the path

        if d >= influence_radius:
            return 0.0

        # Standard robotics repulsion potential (Khatib 1986)
        # Smooth, non-zero gradient everywhere, blows up near surface
        return 0.5 * ((1.0 / d_safe) - (1.0 / influence_radius)) ** 2
    
    def collision_risk_gradient_at(self, point: np.ndarray) -> np.ndarray:
        """Analytical gradient of repulsion potential — no finite differences."""
        influence_radius = 20.0
        best_dist = float("inf")
        best_grad_dir = np.zeros(2)

        for obs in self.obstacles:
            d, grad_dir = obs.distance_and_gradient(point)
            if d < best_dist:
                best_dist = d
                best_grad_dir = grad_dir

        d_safe = max(best_dist, 1e-3)
        if best_dist >= influence_radius:
            return np.zeros(2)

        # ∂U/∂p = -(1/d - 1/d0) * (1/d^2) * ∂d/∂p
        scalar = -((1.0 / d_safe) - (1.0 / influence_radius)) / (d_safe ** 2)
        return scalar * best_grad_dir

class TerrainField:
    """
    Simple rasterized terrain: grid with energy values.
    """
    def __init__(self,
                 width: float,
                 height: float,
                 nx: int,
                 ny: int,
                 energy_grid: np.ndarray):
        """
        energy_grid shape: (ny, nx)
        """
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        self.energy_grid = energy_grid

    def energy_at(self, point: np.ndarray) -> float:
        """
        Bilinear interpolation of terrain energy.
        """
        x, y = point
        # normalize
        gx = np.clip(x / self.width * (self.nx - 1), 0, self.nx - 1)
        gy = np.clip(y / self.height * (self.ny - 1), 0, self.ny - 1)

        x0 = int(np.floor(gx))
        x1 = min(x0 + 1, self.nx - 1)
        y0 = int(np.floor(gy))
        y1 = min(y0 + 1, self.ny - 1)

        wx = gx - x0
        wy = gy - y0

        v00 = self.energy_grid[y0, x0]
        v10 = self.energy_grid[y0, x1]
        v01 = self.energy_grid[y1, x0]
        v11 = self.energy_grid[y1, x1]

        v0 = v00 * (1 - wx) + v10 * wx
        v1 = v01 * (1 - wx) + v11 * wx

        return float(v0 * (1 - wy) + v1 * wy)

class Obstacle(ABC):
    """
    Base class for obstacles. distance_to returns distance to surface.
    Inside obstacle: distance 0.
    """
    @abstractmethod
    def distance_to(self, point: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def distance_and_gradient(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns (distance, unit vector pointing away from obstacle)."""
        pass

class RectObstacle(Obstacle):
    """
    Axis-aligned rectangle obstacle.
    """
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def distance_to(self, point: np.ndarray) -> float:
        x, y = point
        dx = max(self.x_min - x, 0, x - self.x_max)
        dy = max(self.y_min - y, 0, y - self.y_max)
        if dx == 0 and dy == 0:
            return 0.0
        return float(np.hypot(dx, dy))

    def distance_and_gradient(self, point: np.ndarray):
        """Returns (distance, unit vector pointing away from obstacle)."""
        x, y = point
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dy = max(self.y_min - y, 0.0, y - self.y_max)
        dist = float(np.hypot(dx, dy))

        if dist < 1e-9:
            # inside: push toward nearest face
            cx = (self.x_min + self.x_max) / 2
            cy = (self.y_min + self.y_max) / 2
            grad_dir = np.array([x - cx, y - cy])
            norm = np.linalg.norm(grad_dir)
            grad_dir = grad_dir / (norm + 1e-9)
            return 0.0, grad_dir

        gx = -1.0 if (self.x_min - x) > 0 else (1.0 if (x - self.x_max) > 0 else 0.0)
        gy = -1.0 if (self.y_min - y) > 0 else (1.0 if (y - self.y_max) > 0 else 0.0)
        grad_dir = np.array([gx, gy])
        norm = np.linalg.norm(grad_dir)
        grad_dir = grad_dir / (norm + 1e-9)
        return dist, grad_dir

class CircleObstacle(Obstacle):
    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def distance_to(self, point: np.ndarray) -> float:
        d = np.linalg.norm(point - self.center)
        if d <= self.radius:
            return 0.0
        return float(d - self.radius)
    
    def distance_and_gradient(self, point: np.ndarray):
        """Returns (distance, unit vector pointing away from obstacle)."""
        vec = point - self.center
        d = np.linalg.norm(vec)
        if d < 1e-9:
            return 0.0, np.array([1.0, 0.0])  # arbitrary direction if dead center
        dist = float(max(0.0, d - self.radius))
        grad_dir = vec / d   # outward unit vector
        return dist, grad_dir
