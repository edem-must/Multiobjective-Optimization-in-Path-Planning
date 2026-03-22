import time
import threading
import numpy as np
from typing import List, Tuple

from optimization import core as path
from optimization.core import Path, OptimizationHistory
from optimization.core import PathPlanningProblem
from optimization import map_elements as map
from optimization import optimizers as optimize


class AlgorithmResult:
    """
    Container for the result of a single algorithm run.

    Stores the final path, optimization history, and timing information
    so that results from parallel runs can be collected and compared
    after both threads finish.

    Attributes:
        best_path (Path):              Best path found by the algorithm.
        history (OptimizationHistory): Recorded intermediate states.
        elapsed_time (float):          Wall-clock time in seconds.
        algorithm_name (str):          Human-readable label for display.
    """

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.best_path: Path | None = None
        self.history: OptimizationHistory | None = None
        self.elapsed_time: float = 0.0

    def print_summary(self, problem: PathPlanningProblem):
        if self.best_path is None:
            print(f"[{self.algorithm_name}] No result.")
            return

        F     = problem.objective(self.best_path)
        iters = len(self.history.paths) if self.history else 0

        print(f"\n{'─' * 45}")
        print(f"  {self.algorithm_name}")
        print(f"{'─' * 45}")

        if problem.c1 != 0:
            print(f"  L(γ) = {problem.path_length(self.best_path):.4f}  "
                f"(weight C1={problem.c1})")

        if problem.c2 != 0:
            print(f"  E(γ) = {problem.path_energy(self.best_path):.4f}  "
                f"(weight C2={problem.c2})")

        if problem.c3 != 0:
            print(f"  R(γ) = {problem.path_risk(self.best_path):.6f}  "
                f"(weight C3={problem.c3})")

        print(f"  F(γ)       = {F:.4f}  (total objective)")
        print(f"  Iterations = {iters}")
        print(f"  Time       = {self.elapsed_time:.3f} s")
        print(f"{'─' * 45}")


    # def print_summary(self, problem: PathPlanningProblem):
    #     """
    #     Prints a formatted summary of the result to stdout including
    #     all three objective components and total wall-clock time.
    #     """
    #     if self.best_path is None:
    #         print(f"[{self.algorithm_name}] No result.")
    #         return

    #     L = problem.path_length(self.best_path)
    #     E = problem.path_energy(self.best_path)
    #     R = problem.path_risk(self.best_path)
    #     F = problem.objective(self.best_path)
    #     iters = len(self.history.paths) if self.history else 0

    #     print(f"\n{'─' * 45}")
    #     print(f"  {self.algorithm_name}")
    #     print(f"{'─' * 45}")
    #     print(f"  L(γ)        = {L:.4f}  (path length)")
    #     print(f"  E(γ)        = {E:.4f}  (terrain energy)")
    #     print(f"  R(γ)        = {R:.6f}  (collision risk)")
    #     print(f"  F(γ)        = {F:.4f}  (total objective)")
    #     print(f"  Iterations  = {iters}")
    #     print(f"  Time        = {self.elapsed_time:.3f} s")
    #     print(f"{'─' * 45}")

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
            c3=1.0, 
            risk_mode="original"
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

    # def run_gradient(self)-> Tuple[path.Path, path.OptimizationHistory]:
    #     """
    #     Runs the Adam gradient-based optimizer.

    #     Before optimization, inner waypoints are randomly perturbed. This
    #     is necessary because the analytical length gradient (thesis eq. 3.2)
    #     is identically zero on a perfectly straight path (opposing unit
    #     vectors cancel), causing Adam to stall at iteration 1.

    #     Returns:
    #         Tuple (best_path, history) where history contains all intermediate
    #         paths for visualization of the gradient descent workflow.
    #     """
    #     initial = self.create_initial_path()
    #     n = initial.n_points

    #     # Perturb perpendicular to the start→goal direction
    #     # This biases the path toward one side of the obstacle
    #     # rather than sending waypoints randomly in all directions
    #     if self.problem.c3 != 0:
    #         direction = self.goal - self.start
    #         direction /= np.linalg.norm(direction)
    #         perp = np.array([-direction[1], direction[0]])  # 90° rotation

    #         for i in range(1, n - 1):
    #             t = i / (n - 1)
    #             scale = np.sin(t * np.pi)
    #             # Small perpendicular nudge + tiny random component
    #             magnitude = np.random.uniform(2.0, 5.0) + scale # was ±8.0
    #             initial.points[i] += perp * magnitude

    #     initial.points[1:-1, 0] = np.clip(initial.points[1:-1, 0], 0, self.game_map.width)
    #     initial.points[1:-1, 1] = np.clip(initial.points[1:-1, 1], 0, self.game_map.height)

    #     opt = optimize.AdamPathOptimizer(self.problem)
    #     return opt.optimize(initial), opt.history

        #initial = self.create_initial_path()
        ## break straight-line symmetry before Adam starts
        #noise = np.random.uniform(-8.0, 8.0, initial.points[1:-1].shape)
        #initial.points[1:-1] += noise
        #initial.points[1:-1, 0] = np.clip(initial.points[1:-1, 0], 0, self.game_map.width)
        #initial.points[1:-1, 1] = np.clip(initial.points[1:-1, 1], 0, self.game_map.height)
        #opt = optimize.AdamPathOptimizer(self.problem)
        #return opt.optimize(initial), opt.history

    def _run_gradient_into(self, result: AlgorithmResult, n_restarts: int = 5):
        """
        Target function for the gradient thread.

        Runs Adam optimizer n_restarts times with different random perturbations
        and keeps the best result. This partially compensates for Adam's
        sensitivity to initial conditions on non-convex objectives — each
        restart may converge to a different local minimum, and the best one
        is selected by comparing final F(γ) values.

        The perturbation alternates sides of the start→goal axis (even restarts
        go left, odd restarts go right) to ensure both detour options around
        obstacles are explored across restarts.

        Args:
            result:     AlgorithmResult container to write output into.
            n_restarts: Number of independent Adam runs (default 5).
        """
        best_path = None
        best_history = None
        best_value = float("inf")

        direction = self.goal - self.start
        direction /= np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0]])  # 90° rotation

        t_start = time.perf_counter()

        for restart in range(n_restarts):
            initial = self.create_initial_path()
            n = initial.n_points

            if self.problem.c3 != 0:
                # Alternate perturbation side across restarts so that
                # both left and right detours are explored
                sign = 1 if restart % 2 == 0 else -1

                for i in range(1, n - 1):
                    scale = np.sin(i / (n - 1) * np.pi)
                    magnitude = np.random.uniform(2.0, 5.0) * scale
                    initial.points[i] += sign * perp * magnitude

            initial.points[1:-1, 0] = np.clip(initial.points[1:-1, 0], 0, self.game_map.width)
            initial.points[1:-1, 1] = np.clip(initial.points[1:-1, 1], 0, self.game_map.height)

            opt = optimize.AdamPathOptimizer(self.problem)
            candidate = opt.optimize(initial)
            value = self.problem.objective(candidate)

            # Keep this run if it produced a better solution
            if value < best_value or (
                value == best_value and
                len(opt.history.paths) > len(best_history.paths if best_history else [])
            ):
                best_value = value
                best_path = candidate
                best_history = opt.history

        t_end = time.perf_counter()

        result.best_path = best_path
        result.history = best_history
        result.elapsed_time = t_end - t_start


    def _run_pso_into(self, result: AlgorithmResult):
        """
        Target function for the PSO thread.

        Runs PSO and stores the result and elapsed time into the provided
        AlgorithmResult container. Designed to be passed to threading.Thread.

        Args:
            result: AlgorithmResult container to write output into.
        """
        initial = self.create_initial_path()
        opt = optimize.ParticleSwarmOptimizer(self.problem)

        # --- time measurement ---
        t_start = time.perf_counter()
        best_path = opt.optimize(initial)
        t_end = time.perf_counter()
        # ------------------------

        result.best_path = best_path
        result.history = opt.history
        result.elapsed_time = t_end - t_start

    # def run_pso(self) -> Tuple[path.Path, path.OptimizationHistory]:
    #     """
    #     Runs the Particle Swarm Optimizer.

    #     Uses the straight-line path only as a structural template. The
    #     actual initial positions of PSO particles are independently
    #     randomized around the straight line in _random_path_like.

    #     Returns:
    #         Tuple (best_path, history) where history records the global
    #         best path at each iteration for workflow visualization.
    #     """
    #     initial = self.create_initial_path()
    #     opt = optimize.ParticleSwarmOptimizer(self.problem)
    #     best_path = opt.optimize(initial)
    #     return best_path, opt.history

    def run_parallel(self) -> Tuple[AlgorithmResult, AlgorithmResult]:
        """
        Runs both algorithms simultaneously in separate threads.

        Creates two AlgorithmResult containers, launches one thread per
        algorithm, then waits for both to complete. The total wall time
        is approximately max(t_adam, t_pso) rather than their sum.

        Thread safety: each algorithm writes only to its own AlgorithmResult
        container and its own optimizer instance, so no shared state exists
        between threads.

        Returns:
            Tuple (adam_result, pso_result), both fully populated.
        """
        adam_result = AlgorithmResult("Adam Gradient")
        pso_result = AlgorithmResult("Particle Swarm Optimization")

        # Create one thread per algorithm
        adam_thread = threading.Thread(
            target=self._run_gradient_into,
            args=(adam_result,),
            name="Thread-Adam"
        )
        pso_thread = threading.Thread(
            target=self._run_pso_into,
            args=(pso_result,),
            name="Thread-PSO"
        )

        # Measure total wall time across both threads
        wall_start = time.perf_counter()

        print("Starting both algorithms in parallel...")
        adam_thread.start()
        pso_thread.start()

        # Block main thread until both finish
        adam_thread.join()
        pso_thread.join()

        wall_end = time.perf_counter()
        print(f"Both algorithms finished. Total wall time: "
              f"{wall_end - wall_start:.3f} s")

        return adam_result, pso_result
    
    # Keep sequential versions for convenience / debugging
    def run_gradient(self) -> Tuple[Path | None, OptimizationHistory | None]:
        """Sequential gradient run (single-threaded, for debugging)."""
        result = AlgorithmResult("Adam Gradient")
        self._run_gradient_into(result)
        return result.best_path, result.history

    def run_pso(self) -> Tuple[Path | None, OptimizationHistory | None]:
        """Sequential PSO run (single-threaded, for debugging)."""
        result = AlgorithmResult("PSO")
        self._run_pso_into(result)
        return result.best_path, result.history