import time
import matplotlib.pyplot as plt

from optimization.experiment import PathPlanningExperiment
from optimization.render import ExperimentRenderer


def main():
    """
    Entry point for the path planning algorithm comparison experiment.

    Runs Adam and PSO in parallel threads, then produces a 2×2 figure:
      [0,0] Adam path on map       [0,1] PSO path on map
      [1,0] Adam F(γ) curve        [1,1] PSO F(γ) curve

    Timing is measured independently per algorithm (CPU time via
    time.perf_counter) and also as total wall time across both threads.
    """

    exp = PathPlanningExperiment()
    renderer = ExperimentRenderer(exp.game_map)

    # --- Run both algorithms in parallel ---------------------------------
    adam_result, pso_result = exp.run_parallel()

    assert adam_result.best_path is not None and adam_result.history is not None, \
    "Adam optimizer returned no result"
    assert pso_result.best_path is not None and pso_result.history is not None, \
    "PSO optimizer returned no result"

    # --- Print result summaries to console -------------------------------
    adam_result.print_summary(exp.problem)
    pso_result.print_summary(exp.problem)

    # Comparison line
    if adam_result.best_path and pso_result.best_path:
        adam_F = exp.problem.objective(adam_result.best_path)
        pso_F  = exp.problem.objective(pso_result.best_path)
        winner = "Adam" if adam_F < pso_F else "PSO"
        diff   = abs(adam_F - pso_F)
        faster = ("Adam" if adam_result.elapsed_time < pso_result.elapsed_time
                  else "PSO")
        print(f"\n  Best solution : {winner} (ΔF = {diff:.4f})")
        print(f"  Faster        : {faster} "
              f"({min(adam_result.elapsed_time, pso_result.elapsed_time):.3f} s "
              f"vs "
              f"{max(adam_result.elapsed_time, pso_result.elapsed_time):.3f} s)\n")

    # --- Build figure ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Path Optimization Comparison: Adam Gradient vs PSO\n"
        "F(γ) = C1·L(γ) + C2·E(γ) + C3·R(γ)",
        fontsize=13
    )

    # --- Top-left: Adam map ----------------------------------------------
    ax = axes[0, 0]
    renderer.draw_map(ax)
    renderer.draw_history(ax, adam_result.history, color="purple",
                          alpha_start=0.05, alpha_end=0.4, stride=10)
    renderer.draw_path(ax, adam_result.best_path, color="royalblue",
                       lw=2.5, label="Adam final path")
    ax.set_title(
        f"Adam Gradient  |  "
        f"F={exp.problem.objective(adam_result.best_path):.2f}  |  "
        f"t={adam_result.elapsed_time:.3f}s"
    )
    ax.legend(loc="upper left", fontsize=8)

    # --- Top-right: PSO map ----------------------------------------------
    ax = axes[0, 1]
    renderer.draw_map(ax)
    renderer.draw_history(ax, pso_result.history, color="darkorange",
                          alpha_start=0.05, alpha_end=0.4, stride=5)
    renderer.draw_path(ax, pso_result.best_path, color="darkorange",
                       lw=2.5, label="PSO final path")
    ax.set_title(
        f"PSO  |  "
        f"F={exp.problem.objective(pso_result.best_path):.2f}  |  "
        f"t={pso_result.elapsed_time:.3f}s"
    )
    ax.legend(loc="upper left", fontsize=8)

    # --- Bottom-left: Adam convergence -----------------------------------
    ax = axes[1, 0]
    renderer.draw_objective_curve(ax, adam_result.history,
                                   color="royalblue", label="Adam F(γ)")
    ax.set_title(
        f"Adam — Convergence  "
        f"({len(adam_result.history.paths)} iterations, "
        f"{adam_result.elapsed_time:.3f}s)"
    )

    # --- Bottom-right: PSO convergence -----------------------------------
    ax = axes[1, 1]
    renderer.draw_objective_curve(ax, pso_result.history,
                                   color="darkorange", label="PSO F(γ)")
    ax.set_title(
        f"PSO — Convergence  "
        f"({len(pso_result.history.paths)} iterations, "
        f"{pso_result.elapsed_time:.3f}s)"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
