# path_planning/render.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from .map_elements import GameMap, RectObstacle, CircleObstacle
from .core import Path, OptimizationHistory

class ExperimentRenderer:
    def __init__(self, game_map: GameMap):
        self.map = game_map

    def draw_map(self, ax):
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
        pts = path.points
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, lw=lw)
        ax.scatter(pts[0, 0], pts[0, 1], c="green", marker="o", s=50)  # start
        ax.scatter(pts[-1, 0], pts[-1, 1], c="red", marker="x", s=50)  # goal

    def draw_history(self, ax, history: OptimizationHistory,
                     color="purple",
                     alpha_start=0.1,
                     alpha_end=0.6):
        n = len(history.paths)
        if n == 0:
            return
        for i, path in enumerate(history.paths):
            alpha = alpha_start + (alpha_end - alpha_start) * (i / max(1, n - 1))
            self.draw_path(ax, path, color=color, alpha=alpha, lw=1.0)
