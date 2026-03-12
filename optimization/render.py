import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .map_elements import GameMap, RectObstacle, CircleObstacle
from .core import Path, OptimizationHistory


class ExperimentRenderer:
    """
    Handles all visualization for the path planning experiment.

    Responsible for drawing three layers on a matplotlib Axes object:
      1. The map background — terrain energy grid and obstacles
      2. The algorithm workflow — intermediate paths recorded in OptimizationHistory
      3. The final result — the optimized path returned by the algorithm

    This separation of concerns means the renderer does not know anything
    about how the path was computed — it only knows how to draw Path objects
    and GameMap objects. The same renderer instance is reused for both
    the gradient method and PSO plots.

    Attributes:
        map (GameMap): The environment to draw. Used to access terrain
                       grid dimensions and the obstacle list.
    """

    def __init__(self, game_map: GameMap):
        self.map = game_map

    def draw_map(self, ax: Axes):
        """
        Draws the full map background on the given matplotlib Axes.

        Renders two layers:
          - Terrain energy grid as a color-mapped image (imshow), where
            brighter/warmer colors indicate higher energy cost terrain
            (e.g. swamp = high cost, open terrain = low cost).
          - Obstacles drawn as filled black patches on top of the terrain.
            RectObstacle → matplotlib Rectangle patch
            CircleObstacle → matplotlib Circle patch

        The image is flipped vertically (np.flipud) because imshow places
        row 0 at the top, while our coordinate system has y=0 at the bottom.

        Args:
            ax: The matplotlib Axes to draw on.
        """
        terrain = self.map.terrain
        extent = (0, self.map.width, 0, self.map.height)

        # Draw terrain as a background heatmap
        # cmap="YlOrBr": yellow (low energy) to brown (high energy)
        img = ax.imshow(
            np.flipud(terrain.energy_grid),
            extent=extent,
            cmap="YlOrBr",
            alpha=0.6,
            vmin=1.0,
            vmax=3.0
        )

        # Add a colorbar so the reader can interpret terrain energy values
        plt.colorbar(img, ax=ax, fraction=0.03, pad=0.04, label="Terrain energy cost")

        # Draw each obstacle as a filled black shape
        for obs in self.map.obstacles:
            if isinstance(obs, RectObstacle):
                rect = mpatches.Rectangle(
                    (obs.x_min, obs.y_min),
                    obs.x_max - obs.x_min,
                    obs.y_max - obs.y_min,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="black",
                    alpha=0.85,
                    zorder=3   # draw obstacles above terrain but below paths
                )
                ax.add_patch(rect)

            elif isinstance(obs, CircleObstacle):
                circ = mpatches.Circle(
                    tuple(obs.center),
                    obs.radius,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="black",
                    alpha=0.85,
                    zorder=3
                )
                ax.add_patch(circ)

        # Configure axes appearance
        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.height)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def draw_path(self,
                  ax: Axes,
                  path: Path,
                  color: str = "blue",
                  alpha: float = 1.0,
                  lw: float = 2.0,
                  label: str | None = None):
        """
        Draws a single path as a polyline with start and goal markers.

        The path is drawn as connected line segments between consecutive
        waypoints. The start point is marked with a green circle and the
        goal point with a red cross to make them easy to identify.

        Args:
            ax:    The matplotlib Axes to draw on.
            path:  The Path object to render.
            color: Line color (any matplotlib color string or hex code).
            alpha: Opacity of the line (0.0 = transparent, 1.0 = opaque).
            lw:    Line width in points.
            label: Optional legend label for this path.
        """
        pts = path.points

        # Draw the path polyline
        ax.plot(pts[:, 0], pts[:, 1],
                color=color, alpha=alpha, lw=lw,
                label=label, zorder=4)

        # Start marker: green filled circle
        ax.scatter(pts[0, 0], pts[0, 1],
                   c="green", marker="o", s=80,
                   zorder=5, label="Start" if label else None)

        # Goal marker: red cross
        ax.scatter(pts[-1, 0], pts[-1, 1],
                   c="red", marker="X", s=80,
                   zorder=5, label="Goal" if label else None)

    def draw_history(self,
                     ax: Axes,
                     history: OptimizationHistory,
                     color: str = "purple",
                     alpha_start: float = 0.05,
                     alpha_end: float = 0.5,
                     stride: int = 5):
        """
        Draws the algorithm workflow by replaying intermediate paths.

        Paths are drawn from earliest (faint) to latest (more opaque),
        giving a visual impression of how the solution evolved over
        iterations. Drawing all iterations can be slow, so a stride
        parameter skips intermediate frames to keep rendering fast.

        Alpha increases linearly from alpha_start (first iteration) to
        alpha_end (last iteration), creating a "fading trail" effect that
        makes the convergence direction visually clear.

        Args:
            ax:          The matplotlib Axes to draw on.
            history:     OptimizationHistory containing recorded path snapshots.
            color:       Color for all intermediate paths.
            alpha_start: Opacity of the earliest recorded path.
            alpha_end:   Opacity of the latest recorded path.
            stride:      Only draw every nth path to keep rendering fast.
        """
        paths = history.paths[::stride]   # subsample for performance
        n = len(paths)
        if n == 0:
            return

        for i, path in enumerate(paths):
            # Linearly interpolate opacity from faint (early) to visible (late)
            alpha = alpha_start + (alpha_end - alpha_start) * (i / max(1, n - 1))
            pts = path.points
            ax.plot(pts[:, 0], pts[:, 1],
                    color=color, alpha=alpha, lw=0.8,
                    zorder=2)   # draw workflow below the final path

    def draw_objective_curve(self, ax: Axes,
                          history: OptimizationHistory,
                          color: str = "blue",
                          label: str = "F(γ)"):
        """
        Draws the convergence curve of F(γ) over iterations on a log scale.

        A logarithmic Y-axis is used because the objective typically drops
        by several orders of magnitude in early iterations — on a linear
        scale this compresses all fine convergence behavior into a flat line.
        Log scale makes both the initial fast drop and the late slow
        fine-tuning visible simultaneously.
        """
        if not history.objective_values:
            return

        values = history.objective_values

        ax.plot(values, color=color, lw=1.5, label=label)

        # Use log scale only if the value range justifies it
        # (at least one order of magnitude difference between max and min)
        if max(values) > 0 and min(values) > 0:
            value_range = max(values) / min(values)
            if value_range > 10:
                ax.set_yscale("log")
                ax.set_ylabel("F(γ)  [log scale]")
            else:
                ax.set_ylabel("F(γ)")
        else:
            ax.set_ylabel("F(γ)")

        ax.set_xlabel("Iteration")
        ax.set_title("Objective convergence")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)   # "both" shows minor grid on log scale

    # def draw_objective_curve(self, ax: Axes,
    #                           history: OptimizationHistory,
    #                           color: str = "blue",
    #                           label: str = "F(γ)"):
    #     """
    #     Draws the convergence curve of the objective function F(γ) over iterations.

    #     Plots objective value against iteration number on a provided Axes.
    #     This is useful for the results chapter of the thesis to compare
    #     convergence speed between the gradient method and PSO.

    #     Args:
    #         ax:      The matplotlib Axes to draw on (typically a separate subplot).
    #         history: OptimizationHistory whose objective_values list is plotted.
    #         color:   Line color for this algorithm's convergence curve.
    #         label:   Legend label (e.g. "Adam" or "PSO").
    #     """
    #     if not history.objective_values:
    #         return

    #     ax.plot(history.objective_values, color=color, lw=1.5, label=label)
    #     ax.set_xlabel("Iteration")
    #     ax.set_ylabel("F(γ)")
    #     ax.set_title("Objective convergence")
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
