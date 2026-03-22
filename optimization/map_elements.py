import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

class GameMap:
    """
    Represents the full 2D continuous environment for path planning.

    Combines terrain energy information (TerrainField) and obstacles
    (list of Obstacle subclasses) to implement the three sub-objectives
    from the thesis: energy E(γ), and risk R(γ).

    The risk function uses a Khatib-style artificial potential field:
        U(d) = 0.5 * (1/d - 1/d_0)²   for d < d_0
               0                         for d >= d_0

    This formulation (from robotics potential field literature) provides:
      - Smooth repulsion that grows to infinity at d=0
      - Zero influence beyond influence_radius d_0
      - Non-zero gradient everywhere, which is essential for gradient-based optimization

    Attributes:
        width, height (float):          Physical map dimensions.
        terrain (TerrainField):         Terrain energy grid.
        obstacles (List[Obstacle]):     List of all obstacles on the map.
        risk_epsilon (float):           Small constant to avoid division by zero.
        influence_radius (float):       Distance at which obstacle repulsion becomes zero.
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
        """Delegates terrain energy lookup to the TerrainField."""
        return self.terrain.energy_at(point)

    def nearest_obstacle_distance(self, point: np.ndarray) -> float:
        """
        Returns the distance from the point to the surface of the nearest obstacle.
        Returns infinity if there are no obstacles.
        """
        dists = [obs.distance_to(point) for obs in self.obstacles]
        if not dists:
            return float("inf")
        return min(dists)

    def collision_risk_at(self, point: np.ndarray) -> float:
        """
        Evaluates the repulsion potential at a given point.

        Uses the Khatib potential field formulation, which is standard in
        robotics path planning. Points far from all obstacles contribute zero
        risk; points near obstacles are increasingly penalized.

        Args:
            point: [x, y] position to evaluate.

        Returns:
            Scalar risk value >= 0.
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
        """
        Returns the analytical gradient of the repulsion potential at a point.

        Derived from the Khatib potential:
            ∂U/∂p = -(1/d - 1/d_0) * (1/d²) * ∂d/∂p

        where ∂d/∂p is the outward unit normal provided by the nearest
        obstacle's distance_and_gradient() method.

        Using an analytical gradient (rather than finite differences) avoids
        the zero-gradient bug that occurs when both perturbed points land
        inside the obstacle and the difference cancels to zero.

        Args:
            point: [x, y] position to evaluate.

        Returns:
            np.ndarray of shape (2,) — gradient vector [∂U/∂x, ∂U/∂y].
        """
        influence_radius = 20.0
        best_dist = float("inf")
        best_grad_dir = np.zeros(2)

        # Find the nearest obstacle and get its outward direction
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
    
    def collision_risk_original_at(self, point: np.ndarray) -> float:
        """
        Original risk function from thesis eq. 1.4 (pre-Khatib version):

            U(d) = 1 / (||p - t||² + ε)

        where d is the distance to the nearest obstacle and ε is a small
        smoothing constant that prevents division by zero.

        Kept for experimental comparison against the Khatib potential.
        Use PathPlanningProblem(risk_mode="original") to activate this.
        """
        d = self.nearest_obstacle_distance(point)
        return 1.0 / (d ** 2 + self.risk_epsilon)

    def collision_risk_gradient_original_at(self, point: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of the original risk function (thesis eq. 3.4,
        original formulation before replacement with Khatib potential):

            ∂U/∂p = -2(p - t) / (||p - t||² + ε)²

        where (p - t) is the vector from the nearest obstacle surface
        toward the point, provided by distance_and_gradient().

        Note: this gradient is near-zero at practical distances and collapses
        to zero inside obstacles — the main reason the Khatib potential
        was adopted instead. This method exists for experimental testing only.
        """
        best_dist = float("inf")
        best_grad_dir = np.zeros(2)

        for obs in self.obstacles:
            d, grad_dir = obs.distance_and_gradient(point)
            if d < best_dist:
                best_dist = d
                best_grad_dir = grad_dir

        denom = (best_dist ** 2 + self.risk_epsilon) ** 2
        # Chain rule: ∂(1/(d²+ε))/∂p = -2d * (∂d/∂p) / (d²+ε)²
        return -2.0 * best_dist * best_grad_dir / denom

class TerrainField:
    """
    Models terrain energy costs over the 2D map as a discrete grid.

    Corresponds to the energy consumption function E(γ) from the thesis
    (Section 1.2.2), where different terrain types (forest, swamp, desert)
    carry different energy penalties per unit traversed.

    The grid is stored as a 2D array of shape (ny, nx). Values at
    arbitrary continuous positions are retrieved via bilinear interpolation,
    which also makes the energy gradient smooth and differentiable
    (thesis eq. 3.3).

    Attributes:
        width, height (float):       Physical dimensions of the map.
        nx, ny (int):                Grid resolution.
        energy_grid (np.ndarray):    Energy cost at each grid cell, shape (ny, nx).
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
        Returns the terrain energy cost at a continuous 2D position using
        bilinear interpolation across the four nearest grid cells.

        Bilinear interpolation ensures the energy function is smooth,
        which is required for the numerical gradient computation in
        PathPlanningProblem._numerical_gradient_single_point.

        Args:
            point: [x, y] position in map coordinates.

        Returns:
            Interpolated energy cost at the given position.
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
    Abstract base class for all obstacle shapes.

    Each obstacle must implement:
      - distance_to(point): scalar distance from point to obstacle surface
      - distance_and_gradient(point): distance + unit outward normal vector

    The outward gradient direction is used by the analytical risk gradient
    in PathPlanningProblem.risk_gradient_at_waypoint to correctly push
    waypoints away from obstacle surfaces.
    """
    @abstractmethod
    def distance_to(self, point: np.ndarray) -> float:
        """Minimum distance from point to the obstacle surface. Returns 0 if inside."""
        pass
    
    @abstractmethod
    def distance_and_gradient(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Returns both the distance to the obstacle surface and the unit vector
        pointing away from it (outward normal).

        Used by the analytical risk gradient to correctly propagate the
        repulsion force direction without finite differences.

        Returns:
            (distance, outward_unit_vector)
        """
        pass

class RectObstacle(Obstacle):
    """
    Axis-aligned rectangular (box) obstacle.

    Models walls, buildings, or box-shaped terrain blockers.
    Distance is computed as the Euclidean distance to the nearest
    point on the rectangle boundary.

    Args:
        x_min, y_min: Bottom-left corner.
        x_max, y_max: Top-right corner.
    """
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def distance_to(self, point: np.ndarray) -> float:
        """
        Returns 0 if the point is inside the rectangle, otherwise the
        Euclidean distance from the point to the nearest edge or corner.
        """
        x, y = point
        dx = max(self.x_min - x, 0, x - self.x_max)
        dy = max(self.y_min - y, 0, y - self.y_max)
        if dx == 0 and dy == 0:
            return 0.0
        return float(np.hypot(dx, dy))

    def distance_and_gradient(self, point: np.ndarray):
        """
        Returns distance to the rectangle surface and the outward-pointing
        unit normal.

        If the point is inside the obstacle (distance=0), the gradient
        points toward the nearest face by using the vector from the
        rectangle center — this ensures the optimizer always has a
        direction to push the waypoint out.
        """
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
    """
    Circular obstacle defined by center and radius.

    Models round obstacles such as pillars, trees, or craters.
    Distance to surface is simply ||p - center|| - radius.

    Args:
        center: (x, y) tuple of the circle center.
        radius: Radius of the obstacle.
    """
    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def distance_to(self, point: np.ndarray) -> float:
        """
        Returns 0 if inside the circle, otherwise the distance from the
        point to the circle's surface.
        """
        d = np.linalg.norm(point - self.center)
        if d <= self.radius:
            return 0.0
        return float(d - self.radius)
    
    def distance_and_gradient(self, point: np.ndarray):
        """
        Returns distance to circle surface and outward unit normal.

        The outward normal always points radially away from the center,
        making this analytically exact with no edge cases except when
        the point sits exactly at the center (handled with a fallback).
        """
        vec = point - self.center
        d = np.linalg.norm(vec)
        if d < 1e-9:
            return 0.0, np.array([1.0, 0.0])  # arbitrary direction if dead center
        dist = float(max(0.0, d - self.radius))
        grad_dir = vec / d   # outward unit vector
        return dist, grad_dir
