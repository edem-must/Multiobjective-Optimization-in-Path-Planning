# path_planning/render.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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

    def draw_map(self, ax):
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
        # show terrain energy as background image
        extent = [0, self.map.width, 0, self.map.height]
        ax.imshow(
            np.flipud(terrain.energy_grid),
            extent=extent,
            cmap="terrain",
            alpha=0.6
        )

        # draw obstacles
        for obs in self.map.obstacles:
            if isinstance(obs, RectObstacle):
                # rectangle
                rect = Rectangle(
                    (obs.x_min, obs.y_min),
                    obs.x_max - obs.x_min,
                    obs.y_max - obs.y_min,
                    color="black",
                    alpha=0.7
                )
                ax.add_patch(rect)
            elif isinstance(obs, CircleObstacle):
                circ = Circle(
                    tuple(obs.center),
                    obs.radius,
                    color="black",
                    alpha=0.7
                )
                ax.add_patch(circ)

        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.height)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Path Planning Map")

    def draw_path(self, ax, path: Path, color="blue", alpha=1.0, lw=2.0):
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
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, lw=lw)
        ax.scatter(pts[0, 0], pts[0, 1], c="green", marker="o", s=50)  # start
        ax.scatter(pts[-1, 0], pts[-1, 1], c="red", marker="x", s=50)  # goal

    def draw_history(self, ax, history: OptimizationHistory,
                     color="purple",
                     alpha_start=0.1,
                     alpha_end=0.6):
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
        n = len(history.paths)
        if n == 0:
            return
        for i, path in enumerate(history.paths):
            alpha = alpha_start + (alpha_end - alpha_start) * (i / max(1, n - 1))
            self.draw_path(ax, path, color=color, alpha=alpha, lw=1.0)
