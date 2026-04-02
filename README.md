# Multiobjective Optimization in Path Planning

Bachelor's thesis project — comparative analysis of two path optimization algorithms (**Adam** gradient descent and **Particle Swarm Optimization**) on a 2D multi-criteria path planning problem.

The program finds an optimal path between two points on a map with terrain energy costs and geometric obstacles, minimizing a weighted composite objective:

$$F(\gamma) = C_1 \cdot L(\gamma) + C_2 \cdot E(\gamma) + C_3 \cdot R(\gamma)$$

where $L$ is path length, $E$ is terrain energy cost, and $R$ is collision risk.

---

## Repository structure

```
.
├── main.py               # Run a single experiment (one weight combination)
├── grid_search.py        # Exhaustive grid search over C1, C2, C3 values
├── weight_sweep.py       # Sweep over weight combinations and save results
├── optimization/
│   ├── core.py           # Path, PathPlanningProblem, gradients, history
│   ├── map_elements.py   # GameMap, TerrainField, RectObstacle, CircleObstacle
│   ├── optimizers.py     # AdamPathOptimizer, ParticleSwarmOptimizer
│   ├── experiment.py     # PathPlanningExperiment — sets up map and runs both algorithms
│   └── render.py         # ExperimentRenderer — visualization
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/edem-must/Multiobjective-Optimization-in-Path-Planning.git
cd Multiobjective-Optimization-in-Path-Planning
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires **Python 3.10 or newer**.

---

## How to run

### Single experiment — visualize one weight combination

```bash
python main.py
```

This runs both Adam and PSO **in parallel** on the default map with weights `C1=1, C2=1, C3=0` and opens a 2×2 figure:
- Top row: map with terrain, obstacles, path evolution, and final path for each algorithm
- Bottom row: convergence curve of F(γ) over iterations for each algorithm

To change the weights, edit the `c1`, `c2`, `c3` values inside `main.py`:

```python
problem = PathPlanningProblem(
    game_map=experiment.game_map,
    start=experiment.start,
    goal=experiment.goal,
    c1=1.0,   # ← weight for path length
    c2=1.0,   # ← weight for terrain energy
    c3=0.0,   # ← weight for collision risk
)
```

### Grid search — sweep all weight combinations

```bash
python grid_search.py
```

Runs all combinations of `C1, C2, C3 ∈ {0, 0.1, 1, 10, 100}` (excluding the all-zero case) and saves results to `results/grid_search_results.csv`.

> ⚠️ This can take **several hours** depending on your machine. Results are written row-by-row so the file is valid even if the run is interrupted.

### Weight sweep

```bash
python weight_sweep.py
```

A lighter sweep that saves a `results/summary.csv` suitable for thesis analysis and plotting.

---

## Map description

The test map is **100 × 100** units with:

| Element | Description |
|---|---|
| Start point A | (5, 5) |
| Goal point B | (95, 95) |
| Open terrain | Energy cost 1.0 (default) |
| Forest | Energy cost 2.0 — top-left corner (rows 0–20, columns 0–20) |
| Swamp | Energy cost 3.0 — vertical band (columns 20–30) |
| Rectangular obstacle | Corners (40, 40) and (60, 60) — blocks the straight diagonal |
| Circular obstacle | Center (70, 30), radius 8 |

The straight-line path from A to B passes through the rectangular obstacle, forcing the algorithms to find a bypass.

---

## Algorithm overview

### Adam (gradient-based)
- Runs **5 restarts** with different perturbations (alternating left/right of the A→B axis)
- Keeps the best result by final F(γ)
- Early stopping: gradient norm < 10⁻⁶ **or** no improvement > 10⁻⁴ over 20 iterations
- Total time = sum of all restart durations

### PSO (Particle Swarm Optimization)
- **30 particles**, each representing a full candidate path
- Particles initialized as randomly perturbed straight lines (uniform noise ±20)
- Waypoints re-sorted after each update to prevent path self-intersections
- Early stopping: no improvement > 0.1 over 10 iterations

---

## Output files

After running `grid_search.py` or `weight_sweep.py` a `results/` folder is created:

| File | Contents |
|---|---|
| `results/summary.csv` | One row per weight combination: L, E, R, F, time, iters for both algorithms |
| `results/grid_search_results.csv` | Full grid search results |

---

## Dependencies

See `requirements.txt`. All packages are standard:

| Package | Purpose |
|---|---|
| `numpy` | Array operations, gradient computation |
| `matplotlib` | Visualization of paths and convergence curves |
| `pandas` | Saving and loading results CSV files |
| `tqdm` | Progress bar during grid search |
