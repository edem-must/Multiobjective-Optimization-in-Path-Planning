# main.py
import matplotlib.pyplot as plt
from optimization.experiment import PathPlanningExperiment
from optimization.render import ExperimentRenderer

def main():
    exp = PathPlanningExperiment()
    renderer = ExperimentRenderer(exp.game_map)

    gd_path, gd_history = exp.run_gradient()
    pso_path, pso_history = exp.run_pso()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ADAM Gradient descent
    ax = axes[0]
    renderer.draw_map(ax)
    renderer.draw_history(ax, gd_history)
    renderer.draw_path(ax, gd_path, color="blue", alpha=1.0, lw=2.5)
    ax.set_title("ADAM Gradient Descent")

    # PSO
    ax = axes[1]
    renderer.draw_map(ax)
    renderer.draw_history(ax, pso_history)
    renderer.draw_path(ax, pso_path, color="orange", alpha=1.0, lw=2.5)
    ax.set_title("Particle Swarm Optimization")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
