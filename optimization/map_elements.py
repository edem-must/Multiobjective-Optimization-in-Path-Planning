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
                 risk_epsilon: float = 0.1):
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
        return 1.0 / (d * d + self.risk_epsilon)

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

class CircleObstacle(Obstacle):
    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def distance_to(self, point: np.ndarray) -> float:
        d = np.linalg.norm(point - self.center)
        if d <= self.radius:
            return 0.0
        return float(d - self.radius)
