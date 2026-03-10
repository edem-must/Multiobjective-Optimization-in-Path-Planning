# main.py
import matplotlib.pyplot as plt
from optimization.experiment import PathPlanningExperiment
from optimization.render import ExperimentRenderer

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
    gd_path, gd_history = exp.run_gradient()

    print("Running Particle Swarm Optimizer...")
    pso_path, pso_history = exp.run_pso()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Top-left: Adam path on map --------------------------------------
    ax = axes[0]
    renderer.draw_map(ax)
    renderer.draw_history(ax, gd_history)
    renderer.draw_path(ax, gd_path, color="blue", alpha=1.0, lw=2.5)
    ax.set_title("ADAM Gradient Descent")

    # --- Top-right: PSO path on map --------------------------------------
    ax = axes[1]
    renderer.draw_map(ax)
    renderer.draw_history(ax, pso_history)
    renderer.draw_path(ax, pso_path, color="orange", alpha=1.0, lw=2.5)
    ax.set_title("Particle Swarm Optimization")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Guard ensures main() is only called when running this file directly,
    # not when it is imported as a module from elsewhere.
    main()
