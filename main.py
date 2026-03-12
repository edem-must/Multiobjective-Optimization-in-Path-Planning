# main.py
import matplotlib.pyplot as plt
from optimization.experiment import PathPlanningExperiment
from optimization.render import ExperimentRenderer

#import matplotlib.pyplot as plt
#import numpy as np

#from path_planning.experiment import PathPlanningExperiment
#from path_planning.render import ExperimentRenderer


def main():
    """
    Entry point for the path planning algorithm comparison experiment.

    Runs both the gradient-based (Adam) optimizer and the Particle Swarm
    Optimizer on the same test map, then produces a side-by-side visualization
    showing:
      - Left column:  Adam gradient method — map, workflow, final path
      - Right column: PSO — map, workflow, final path
      - Bottom row:   Convergence curves of F(γ) for both algorithms

    The figure layout is (2 rows × 2 columns):
      [0,0] Adam path on map    [0,1] PSO path on map
      [1,0] Adam F(γ) curve     [1,1] PSO F(γ) curve

    Output is displayed interactively via plt.show().
    No files are written; add plt.savefig() if you need to export figures
    for your thesis document.
    """

    # --- Setup -----------------------------------------------------------
    # Create the experiment: builds the default map and problem instance
    exp = PathPlanningExperiment()

    # Create the renderer using the same map so drawings are consistent
    renderer = ExperimentRenderer(exp.game_map)

    # --- Run algorithms --------------------------------------------------
    # Both calls return (best_path, history).
    # History records intermediate paths for workflow visualization.
    print("Running Adam gradient optimizer...")
    adam_path, adam_history = exp.run_gradient()
    print(f"  Done. Iterations recorded: {len(adam_history.paths)}")
    print(f"  Final objective F(γ) = {adam_history.objective_values[-1]:.4f}")

    print("Running Particle Swarm Optimizer...")
    pso_path, pso_history = exp.run_pso()
    print(f"  Done. Iterations recorded: {len(pso_history.paths)}")
    print(f"  Final objective F(γ) = {pso_history.objective_values[-1]:.4f}")

    # --- Print component breakdown ---------------------------------------
    # Decompose F(γ) into its three sub-objectives for analysis
    print("\n--- Result breakdown ---")
    for name, path in [("Adam", adam_path), ("PSO", pso_path)]:
        L = exp.problem.path_length(path)
        E = exp.problem.path_energy(path)
        R = exp.problem.path_risk(path)
        F = exp.problem.objective(path)
        print(f"{name}:  L={L:.2f}  E={E:.2f}  R={R:.4f}  F={F:.4f}")

    # --- Build figure layout ---------------------------------------------
    # 2×2 grid: top row = path maps, bottom row = convergence curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Path Optimization Comparison: Adam Gradient vs PSO\n"
        "F(γ) = C1·L(γ) + C2·E(γ) + C3·R(γ)",
        fontsize=13
    )

    # --- Top-left: Adam path on map --------------------------------------
    ax = axes[0, 0]
    renderer.draw_map(ax)

    # Draw intermediate paths as a faint purple workflow trail
    renderer.draw_history(ax, adam_history, color="purple",
                          alpha_start=0.05, alpha_end=0.4, stride=10)

    # Draw final optimized path in blue on top
    renderer.draw_path(ax, adam_path, color="royalblue", lw=2.5,
                       label="Adam final path")

    ax.set_title("Adam Gradient Optimizer")
    ax.legend(loc="upper left", fontsize=8)

    # --- Top-right: PSO path on map --------------------------------------
    ax = axes[0, 1]
    renderer.draw_map(ax)

    # Draw swarm history — shows particles converging toward good solutions
    renderer.draw_history(ax, pso_history, color="darkorange",
                          alpha_start=0.05, alpha_end=0.4, stride=5)

    # Draw final best path found by PSO in orange
    renderer.draw_path(ax, pso_path, color="darkorange", lw=2.5,
                       label="PSO final path")

    ax.set_title("Particle Swarm Optimization")
    ax.legend(loc="upper left", fontsize=8)

    # --- Bottom-left: Adam convergence curve -----------------------------
    ax = axes[1, 0]
    renderer.draw_objective_curve(ax, adam_history,
                                   color="royalblue", label="Adam F(γ)")
    ax.set_title("Adam — Objective convergence")

    # --- Bottom-right: PSO convergence curve -----------------------------
    ax = axes[1, 1]
    renderer.draw_objective_curve(ax, pso_history,
                                   color="darkorange", label="PSO F(γ)")
    ax.set_title("PSO — Objective convergence")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Guard ensures main() is only called when running this file directly,
    # not when it is imported as a module from elsewhere.
    main()
