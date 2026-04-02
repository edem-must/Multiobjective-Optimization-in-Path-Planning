# weight_sweep.py
"""
Batch sweep over all combinations of C1, C2, C3 ∈ {0, 0.1, 1, 10, 100}.
Both Adam and PSO run in parallel per combination (same as run_parallel).

Output structure:
    results/
        single_criterion/    ← exactly 1 non-zero weight
        double_criteria/     ← exactly 2 non-zero weights
        multi_criteria/      ← all 3 non-zero weights
        summary.csv          ← all numeric results in one file

Estimated runtime: ~124 combinations × ~90s each ≈ 3 hours.
Use WEIGHT_VALUES to narrow the sweep during development.
"""

import csv
import itertools
import threading
import time
from pathlib import Path as FilePath
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive: required for batch saving
import matplotlib.pyplot as plt
import numpy as np

from optimization.core import PathPlanningProblem
from optimization.experiment import AlgorithmResult, PathPlanningExperiment
from optimization.render import ExperimentRenderer

# ──────────────────────────────────────────────
# Configuration — edit these before running
# ──────────────────────────────────────────────
WEIGHT_VALUES: List[float] = [0.0, 0.1, 1.0, 10.0, 100.0]
OUTPUT_ROOT = FilePath("results_2")
ADAM_RESTARTS = 5   # reduce to 1 for quick testing, 5 for thesis runs


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _category(c1: float, c2: float, c3: float) -> str | None:
    n = sum(w != 0.0 for w in (c1, c2, c3))
    if n == 0:
        return None
    return {1: "single_criterion", 2: "double_criteria", 3: "multi_criteria"}[n]


def _tag(c1: float, c2: float, c3: float) -> str:
    """Filename-safe label, e.g. C1-1_C2-0-1_C3-10"""
    def fmt(w: float) -> str:
        s = f"{w:g}".replace(".", "-")
        return s
    return f"C1-{fmt(c1)}_C2-{fmt(c2)}_C3-{fmt(c3)}"


def _save_plot(
    exp: PathPlanningExperiment,
    adam: AlgorithmResult,
    pso: AlgorithmResult,
    path: FilePath,
    c1: float, c2: float, c3: float,
) -> None:
    renderer = ExperimentRenderer(exp.game_map)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_am, ax_pm, ax_ac, ax_pc = axes.flat

    fig.suptitle(
        f"C1={c1}  C2={c2}  C3={c3}  —  "
        f"F(γ) = C1·L(γ) + C2·E(γ) + C3·R(γ)",
        fontsize=11,
    )

    for ax, result, color, label in [
        (ax_am, adam, "blue",       "Adam final path"),
        (ax_pm, pso,  "darkorange", "PSO final path"),
    ]:
        renderer.draw_map(ax)
        if result.history:
            renderer.draw_history(ax, result.history, color=color)
        if result.best_path:
            F = exp.problem.objective(result.best_path)
            renderer.draw_path(ax, result.best_path, color=color, label=label)
            ax.set_title(
                f"{result.algorithm_name}  |  F={F:.2f}  |  t={result.elapsed_time:.3f}s",
                fontsize=9,
            )
        ax.legend(loc="upper left", fontsize=8)

    for ax, result, color, label in [
        (ax_ac, adam, "blue",       "Adam F(γ)"),
        (ax_pc, pso,  "darkorange", "PSO F(γ)"),
    ]:
        if result.history:
            renderer.draw_objective_curve(ax, result.history, color=color, label=label)
            iters = result.total_iters or len(result.history.paths)
            ax.set_title(
                f"{result.algorithm_name} — Convergence "
                f"({iters} iters, {result.elapsed_time:.3f}s)",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _collect_metrics(result: AlgorithmResult, prob: PathPlanningProblem) -> dict:
    if result.best_path is None:
        return dict(L=None, E=None, R=None, F=None, iters=None, time_s=None)
    return dict(
        L=round(float(prob.path_length(result.best_path)), 4),
        E=round(float(prob.path_energy(result.best_path)), 4),
        R=round(float(prob.path_risk(result.best_path)), 6),
        F=round(float(prob.objective(result.best_path)), 4),
        iters=result.total_iters or len(result.history.paths if result.history else []),
        time_s=round(result.elapsed_time, 3),
    )


# ──────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────

CSV_FIELDS = [
    "category", "c1", "c2", "c3",
    "adam_L", "adam_E", "adam_R", "adam_F", "adam_iters", "adam_time_s",
    "pso_L",  "pso_E",  "pso_R",  "pso_F",  "pso_iters",  "pso_time_s",
]


def run_sweep(
    weight_values: List[float] = WEIGHT_VALUES,
    output_root: FilePath = OUTPUT_ROOT,
    adam_restarts: int = ADAM_RESTARTS,
) -> None:
    # Create output folders
    for cat in ("single_criterion", "double_criteria", "multi_criteria"):
        (output_root / cat).mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "summary.csv"

    # Write CSV header once
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    # Build combination list — skip all-zero
    combos = [
        (c1, c2, c3)
        for c1, c2, c3 in itertools.product(weight_values, repeat=3)
        if _category(c1, c2, c3) is not None
    ]
    total = len(combos)
    print(f"Starting sweep: {total} combinations, {adam_restarts} Adam restarts each.\n")

    sweep_start = time.perf_counter()

    for idx, (c1, c2, c3) in enumerate(combos, start=1):
        category = _category(c1, c2, c3)
        assert category is not None
        tag = _tag(c1, c2, c3)
        print(f"[{idx:>3}/{total}]  {tag}  ({category})")

        # Build experiment and override weights
        exp = PathPlanningExperiment()
        exp.problem.c1 = c1
        exp.problem.c2 = c2
        exp.problem.c3 = c3

        # Run Adam and PSO in parallel (mirrors run_parallel)
        adam_result = AlgorithmResult("Adam Gradient")
        pso_result  = AlgorithmResult("PSO")

        adam_thread = threading.Thread(
            target=exp._run_gradient_into,
            args=(adam_result,),
            kwargs={"n_restarts": adam_restarts},
        )
        pso_thread = threading.Thread(
            target=exp._run_pso_into,
            args=(pso_result,),
        )

        adam_thread.start()
        pso_thread.start()
        adam_thread.join()
        pso_thread.join()

        adam_result.print_summary(exp.problem)
        pso_result.print_summary(exp.problem)

        # Save plot
        plot_path = output_root / category / f"{tag}.png"
        _save_plot(exp, adam_result, pso_result, plot_path, c1, c2, c3)
        print(f"  → {plot_path}")

        # Append CSV row
        am = _collect_metrics(adam_result, exp.problem)
        pm = _collect_metrics(pso_result,  exp.problem)
        row = {
            "category": category, "c1": c1, "c2": c2, "c3": c3,
            "adam_L": am["L"], "adam_E": am["E"], "adam_R": am["R"],
            "adam_F": am["F"], "adam_iters": am["iters"], "adam_time_s": am["time_s"],
            "pso_L":  pm["L"], "pso_E":  pm["E"], "pso_R":  pm["R"],
            "pso_F":  pm["F"], "pso_iters":  pm["iters"], "pso_time_s":  pm["time_s"],
        }
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

    elapsed = time.perf_counter() - sweep_start
    print(f"\n{'='*52}")
    print(f" Sweep done  —  {total} combinations in {elapsed/60:.1f} min")
    print(f" CSV  →  {csv_path}")
    print(f"{'='*52}")


if __name__ == "__main__":
    run_sweep()