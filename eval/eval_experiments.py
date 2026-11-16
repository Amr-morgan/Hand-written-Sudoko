"""
Formal Evaluation Experiments Script for P7 Project Deliverables.
Fulfills university requirements:
- Experiment 1: Search Algorithm Comparison (UCS vs Greedy vs A*)
- Experiment 2: CSP Techniques Comparison (BT vs BT+FC vs BT+FC+AC-3)
- Experiment 3: System Robustness across Misread Injections (0 to 3 misreads)
"""

import os
import sys
import time
import numpy as np

# Add project root directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.generator import SudokuGenerator, MosaicGenerator
from integration.pipeline import SudokuPipeline
from csp.solver import SudokuCSP


def run_experiment_1_search(pipeline, cell_imgs_corrupted):
    print("\n=======================================================")
    print("      EXPERIMENT 1: SEARCH ALGORITHM COMPARISON        ")
    print("=======================================================")
    print(f"{'Algorithm':<12} | {'Solved':<8} | {'Nodes Expanded':<16} | {'Nodes Generated':<16}")
    print("-" * 65)

    for algo in ["UCS", "Greedy", "AStar"]:
        res = pipeline.solve_mosaic(cell_imgs_corrupted, search_algo=algo, csp_mode="BT_FC_AC3")
        print(f"{algo:<12} | {str(res['success']):<8} | {res['search_nodes_expanded']:<16} | {res['search_nodes_generated']:<16}")


def run_experiment_2_csp(puzzle_grid):
    print("\n=======================================================")
    print("       EXPERIMENT 2: CSP SOLVER TECHNIQUES            ")
    print("=======================================================")
    print(f"{'CSP Mode':<15} | {'Solvable':<10} | {'Assignments':<12} | {'Backtracks':<10} | {'Time (s)':<10}")
    print("-" * 68)

    for mode in ["BT", "BT_FC", "BT_FC_AC3"]:
        csp = SudokuCSP(puzzle_grid)
        start_time = time.time()
        solvable, _ = csp.solve(mode=mode)
        elapsed = time.time() - start_time
        print(f"{mode:<15} | {str(solvable):<10} | {csp.assignments:<12} | {csp.backtracks:<10} | {elapsed:<10.4f}")


def run_experiment_3_noise_robustness(pipeline, puzzle_grid, cell_images):
    print("\n=======================================================")
    print("      EXPERIMENT 3: MISREAD NOISE ROBUSTNESS           ")
    print("=======================================================")
    print(f"{'Misreads Injected':<18} | {'Pipeline Success':<18} | {'Corrections Applied':<20}")
    print("-" * 65)

    mosaic_gen = MosaicGenerator()
    for misreads in range(0, 4):
        corrupted_imgs, records = mosaic_gen.inject_misreads(puzzle_grid, cell_images, num_misreads=misreads)
        res = pipeline.solve_mosaic(corrupted_imgs, search_algo="UCS", csp_mode="BT_FC_AC3")
        num_applied = len(res["corrections"])
        print(f"{misreads:<18} | {str(res['success']):<18} | {num_applied:<20}")


def run_all_experiments():
    print("--- Starting Step 12: Automated Project Experiments ---")

    # Generate baseline test puzzle and mosaic
    sudoku_gen = SudokuGenerator()
    _, puzzle_grid = sudoku_gen.generate_puzzle(num_givens=30)
    mosaic_gen = MosaicGenerator()
    _, cell_images = mosaic_gen.create_mosaic(puzzle_grid)

    corrupted_images, _ = mosaic_gen.inject_misreads(puzzle_grid, cell_images, num_misreads=2)

    pipeline = SudokuPipeline()

    run_experiment_1_search(pipeline, corrupted_images)
    run_experiment_2_csp(puzzle_grid)
    run_experiment_3_noise_robustness(pipeline, puzzle_grid, cell_images)

    print("\n🎉 Step 12 completed successfully!")


if __name__ == "__main__":
    run_all_experiments()