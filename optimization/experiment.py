import numpy as np
from typing import List, Tuple
from optimization import core as path
import optimization.map_elements as map
import optimization.optimizers as optimize

class PathPlanningExperiment:
    """
    High-level experiment runner for comparing path optimization algorithms.

    Sets up a fixed test scenario (map + problem) and provides a uniform
    interface to run each algorithm and collect results. This corresponds
    to the methodology described in Chapter 4 of the thesis, where a
    single test environment is used for controlled comparison.

    The default map includes:
      - A swamp band (high energy cost) running vertically through the center
      - A forest region in the top-left corner (medium energy cost)
      - A large rectangular obstacle in the map center
      - A circular obstacle in the lower-right quadrant

    Objective weights are set to give the risk term enough influence to
    force meaningful obstacle avoidance against the competing length term.

    Attributes:
        game_map (GameMap):               The constructed test environment.
        start (np.ndarray):               Start point A.
        goal (np.ndarray):                Goal point B.
        problem (PathPlanningProblem):    The configured optimization problem.
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
            c2=0.0,
            c3=0.0
        )

    def _create_default_map(self) -> map.GameMap:
        """
        Constructs the default test map with terrain and obstacles.

        Terrain is a 50×50 grid where:
          - Default energy = 1.0 (open terrain)
          - Swamp band (columns 20–30) = 3.0
          - Forest region (rows 0–20, columns 0–20) = 2.0

        Obstacles include one rectangle and one circle, positioned so that
        a naive straight-line path from start to goal passes through them.

        Returns:
            Fully configured GameMap instance.
        """
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
        Creates a straight-line path between start and goal.

        This serves as the structural template for PSO particle initialization
        and as the starting point (after perturbation) for the gradient method.

        Args:
            n_points: Number of waypoints including start and goal.

        Returns:
            Straight-line Path from start to goal.
        """
        xs = np.linspace(self.start[0], self.goal[0], n_points)
        ys = np.linspace(self.start[1], self.goal[1], n_points)
        points = np.stack([xs, ys], axis=1)
        return path.Path(points)

    def run_gradient(self)-> Tuple[path.Path, path.OptimizationHistory]:
        """
        Runs the Adam gradient-based optimizer.

        Before optimization, inner waypoints are randomly perturbed. This
        is necessary because the analytical length gradient (thesis eq. 3.2)
        is identically zero on a perfectly straight path (opposing unit
        vectors cancel), causing Adam to stall at iteration 1.

        Returns:
            Tuple (best_path, history) where history contains all intermediate
            paths for visualization of the gradient descent workflow.
        """
        initial = self.create_initial_path()
        # break straight-line symmetry before Adam starts
        noise = np.random.uniform(-8.0, 8.0, initial.points[1:-1].shape)
        initial.points[1:-1] += noise
        initial.points[1:-1, 0] = np.clip(initial.points[1:-1, 0], 0, self.game_map.width)
        initial.points[1:-1, 1] = np.clip(initial.points[1:-1, 1], 0, self.game_map.height)
        opt = optimize.AdamPathOptimizer(self.problem)
        return opt.optimize(initial), opt.history


    def run_pso(self) -> Tuple[path.Path, path.OptimizationHistory]:
        """
        Runs the Particle Swarm Optimizer.

        Uses the straight-line path only as a structural template. The
        actual initial positions of PSO particles are independently
        randomized around the straight line in _random_path_like.

        Returns:
            Tuple (best_path, history) where history records the global
            best path at each iteration for workflow visualization.
        """
        initial = self.create_initial_path()
        opt = optimize.ParticleSwarmOptimizer(self.problem)
        best_path = opt.optimize(initial)
        return best_path, opt.history
