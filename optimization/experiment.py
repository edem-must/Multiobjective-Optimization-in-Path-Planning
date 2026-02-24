import numpy as np
from typing import List, Tuple
from optimization import core as path
import optimization.map_elements as map
import optimization.optimizers as optimize

class PathPlanningExperiment:
    """
    Sets up a single test map, builds problem, runs both algorithms,
    and returns results for visualization.
    """
    def __init__(self):
        self.game_map = self._create_default_map()
        self.start = np.array([5.0, 5.0])
        self.goal = np.array([95.0, 95.0])

        self.problem = path.PathPlanningProblem(
            game_map=self.game_map,
            start=self.start,
            goal=self.goal,
            c1=1.0,
            c2=0.5,
            c3=0.5
        )

    def _create_default_map(self) -> map.GameMap:
        width, height = 100.0, 100.0
        nx, ny = 50, 50

        # Example: base cost 1, swamp stripe, forest stripe, etc.
        energy_grid = np.ones((ny, nx), dtype=float)

        # make a "swamp" vertical band in middle (higher cost)
        energy_grid[:, 20:30] = 3.0

        # "forest" in top-left
        energy_grid[0:20, 0:20] = 2.0

        terrain = map.TerrainField(width, height, nx, ny, energy_grid)

        obstacles: List[map.Obstacle] = [
            map.RectObstacle(40, 40, 60, 60),
            map.CircleObstacle((70, 30), 8.0),
        ]

        return map.GameMap(width, height, terrain, obstacles)

    def create_initial_path(self, n_points: int = 20) -> path.Path:
        """
        Simple initial path: straight line with evenly spaced points.
        """
        xs = np.linspace(self.start[0], self.goal[0], n_points)
        ys = np.linspace(self.start[1], self.goal[1], n_points)
        points = np.stack([xs, ys], axis=1)
        return path.Path(points)

    def run_gradient(self) -> Tuple[path.Path, path.OptimizationHistory]:
        initial = self.create_initial_path()
        opt = optimize.AdamPathOptimizer(self.problem)
        best_path = opt.optimize(initial)
        return best_path, opt.history

    def run_pso(self) -> Tuple[path.Path, path.OptimizationHistory]:
        initial = self.create_initial_path()
        opt = optimize.ParticleSwarmOptimizer(self.problem)
        best_path = opt.optimize(initial)
        return best_path, opt.history
