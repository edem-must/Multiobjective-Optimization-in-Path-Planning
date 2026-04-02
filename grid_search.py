import csv
import time
import itertools
import threading
import numpy as np
from pathlib import Path as FilePath

from optimization import core as path
from optimization.core import PathPlanningProblem
from optimization import map_elements as map_mod
from optimization import optimizers as optimize

# ─────────────────────────────────────────────
#  Configuration — change these freely
# ─────────────────────────────────────────────
STEP          = 0.25    # grid step for c1, c2, c3 (0.25 → 124 combos)
OUTPUT_FILE   = "grid_search_results.csv"

# Reduced parameters for fast grid-search runs
# (increase for higher quality at the cost of time)
ADAM_MAX_ITERS  = 500
ADAM_RESTARTS   = 2
PSO_N_PARTICLES = 20
PSO_MAX_ITERS   = 100
# ─────────────────────────────────────────────


def build_map() -> map_mod.GameMap:
    """
    Constructs the standard test map used in the thesis experiments.
    Identical to PathPlanningExperiment._create_default_map().
    """
    width, height = 100.0, 100.0
    nx, ny = 50, 50

    energy_grid = np.ones((ny, nx), dtype=float)
    energy_grid[:, 20:30] = 3.0   # swamp band
    energy_grid[0:20, 0:20] = 2.0 # forest region

    terrain = map_mod.TerrainField(width, height, nx, ny, energy_grid)
    obstacles = [
        map_mod.RectObstacle(40, 40, 60, 60),
        map_mod.CircleObstacle((70, 30), 8.0),
    ]
    return map_mod.GameMap(width, height, terrain, obstacles)


def create_straight_path(start: np.ndarray,
                          goal: np.ndarray,
                          n_points: int = 20) -> path.Path:
    """Returns a straight-line path from start to goal."""
    xs = np.linspace(start[0], goal[0], n_points)
    ys = np.linspace(start[1], goal[1], n_points)
    return path.Path(np.stack([xs, ys], axis=1))


def run_adam(problem: PathPlanningProblem,
             start: np.ndarray,
             goal: np.ndarray) -> dict:
    """
    Runs Adam optimizer with reduced parameters suitable for grid search.

    Executes ADAM_RESTARTS independent runs with perpendicular perturbation
    (or no perturbation when c3=0) and returns the best result.

    Returns:
        Dict with keys: L, E, R, F, time
    """
    best_value = float("inf")
    best_path_obj = None

    direction = goal - start
    direction /= np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])

    t_start = time.perf_counter()

    for restart in range(ADAM_RESTARTS):
        initial = create_straight_path(start, goal)
        n = initial.n_points

        if problem.c3 != 0:
            sign = 1 if restart % 2 == 0 else -1
            for i in range(1, n - 1):
                scale = np.sin(i / (n - 1) * np.pi)
                magnitude = np.random.uniform(2.0, 5.0) * scale
                initial.points[i] += sign * perp * magnitude

        initial.points[1:-1, 0] = np.clip(
            initial.points[1:-1, 0], 0, problem.map.width)
        initial.points[1:-1, 1] = np.clip(
            initial.points[1:-1, 1], 0, problem.map.height)

        opt = optimize.AdamPathOptimizer(
            problem,
            max_iters=ADAM_MAX_ITERS
        )
        candidate = opt.optimize(initial)
        val = problem.objective(candidate)

        if val < best_value:
            best_value = val
            best_path_obj = candidate

    t_end = time.perf_counter()

    assert best_path_obj is not None, \
    "Adam optimizer returned no result"

    return {
        "L":    round(problem.path_length(best_path_obj), 6),
        "E":    round(problem.path_energy(best_path_obj), 6),
        "R":    round(problem.path_risk(best_path_obj), 6),
        "F":    round(best_value, 6),
        "time": round(t_end - t_start, 4),
    }


def run_pso(problem: PathPlanningProblem,
            start: np.ndarray,
            goal: np.ndarray) -> dict:
    """
    Runs PSO with reduced parameters suitable for grid search.

    Returns:
        Dict with keys: L, E, R, F, time
    """
    initial = create_straight_path(start, goal)

    opt = optimize.ParticleSwarmOptimizer(
        problem,
        n_particles=PSO_N_PARTICLES,
        max_iters=PSO_MAX_ITERS
    )

    t_start = time.perf_counter()
    best_path_obj = opt.optimize(initial)
    t_end = time.perf_counter()

    return {
        "L":    round(problem.path_length(best_path_obj), 6),
        "E":    round(problem.path_energy(best_path_obj), 6),
        "R":    round(problem.path_risk(best_path_obj), 6),
        "F":    round(problem.objective(best_path_obj), 6),
        "time": round(t_end - t_start, 4),
    }


def run_combo_parallel(game_map, start, goal, c1, c2, c3) -> dict:
    """
    Runs Adam and PSO in parallel threads for one weight combination.

    Returns a single flat dict row ready to be written to CSV.
    """
    problem = PathPlanningProblem(
        game_map=game_map,
        start=start,
        goal=goal,
        c1=c1, c2=c2, c3=c3
    )

    adam_result = {}
    pso_result  = {}

    adam_thread = threading.Thread(
        target=lambda: adam_result.update(run_adam(problem, start, goal))
    )
    pso_thread = threading.Thread(
        target=lambda: pso_result.update(run_pso(problem, start, goal))
    )

    adam_thread.start()
    pso_thread.start()
    adam_thread.join()
    pso_thread.join()

    return {
        "c1": c1, "c2": c2, "c3": c3,
        # Adam components
        "adam_L":    adam_result["L"],
        "adam_E":    adam_result["E"],
        "adam_R":    adam_result["R"],
        "adam_F":    adam_result["F"],
        "adam_time": adam_result["time"],
        # PSO components
        "pso_L":    pso_result["L"],
        "pso_E":    pso_result["E"],
        "pso_R":    pso_result["R"],
        "pso_F":    pso_result["F"],
        "pso_time": pso_result["time"],
        # Derived: which algorithm won and by how much
        "winner":   "Adam" if adam_result["F"] <= pso_result["F"] else "PSO",
        "delta_F":  round(abs(adam_result["F"] - pso_result["F"]), 6),
    }


def generate_weight_combinations(step: float) -> list:
    """
    Generates all combinations of (c1, c2, c3) in [0, 1] at given step size,
    excluding (0, 0, 0) since it produces a trivially empty objective.

    Args:
        step: Grid spacing, e.g. 0.25 → values [0.0, 0.25, 0.5, 0.75, 1.0]

    Returns:
        List of (c1, c2, c3) tuples.
    """
    values = [round(v, 10) for v in np.arange(0.0, 1.0 + step / 2, step)]
    combos = list(itertools.product(values, repeat=3))
    # Remove the all-zero combination
    combos = [(c1, c2, c3) for (c1, c2, c3) in combos
              if not (c1 == 0 and c2 == 0 and c3 == 0)]
    return combos


def main():
    game_map = build_map()
    start    = np.array([5.0, 5.0])
    goal     = np.array([95.0, 95.0])

    combos = generate_weight_combinations(STEP)
    total  = len(combos)

    print(f"Grid search: {total} weight combinations")
    print(f"  Step size        : {STEP}")
    print(f"  Adam restarts    : {ADAM_RESTARTS},  max_iters={ADAM_MAX_ITERS}")
    print(f"  PSO particles    : {PSO_N_PARTICLES}, max_iters={PSO_MAX_ITERS}")
    print(f"  Output file      : {OUTPUT_FILE}")
    print(f"  Estimated time   : ~{total * 3 // 60} min\n")

    # CSV column order
    fieldnames = [
        "c1", "c2", "c3",
        "adam_L", "adam_E", "adam_R", "adam_F", "adam_time",
        "pso_L",  "pso_E",  "pso_R",  "pso_F",  "pso_time",
        "winner", "delta_F"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (c1, c2, c3) in enumerate(combos, start=1):
            t0 = time.perf_counter()

            row = run_combo_parallel(game_map, start, goal, c1, c2, c3)
            writer.writerow(row)
            f.flush()  # write each row immediately so data is safe on crash

            elapsed = time.perf_counter() - t0
            remaining = (total - idx) * elapsed
            print(
                f"[{idx:>4}/{total}]  "
                f"c1={c1:.2f} c2={c2:.2f} c3={c3:.2f}  |  "
                f"Adam F={row['adam_F']:.2f}  PSO F={row['pso_F']:.2f}  "
                f"winner={row['winner']}  "
                f"({elapsed:.1f}s, ~{remaining/60:.1f} min left)"
            )

    print(f"\nDone. Results saved to: {FilePath(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    main()