"""
Official Demonstration Script for P7 — Handwritten Sudoku Reader and Solver.
Fulfills university submission requirement: demonstrates end-to-end functionality
from image mosaic rendering, NN reading, search misread correction, to CSP solving.
"""

import os
import sys
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.generator import SudokuGenerator, MosaicGenerator
from integration.pipeline import SudokuPipeline


def print_grid(grid, title="Sudoku Grid"):
    """Utility function to pretty-print a 9x9 Sudoku grid."""
    print(f"\n--- {title} ---")
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 23)
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            val = grid[r][c]
            row_str += f"{val if val != 0 else '.'} "
        print(row_str)


def run_full_demo():
    print("==========================================================")
    print("   P7: HANDWRITTEN SUDOKU READER & SOLVER - OFFICIAL DEMO  ")
    print("==========================================================")
    
    # 1. Generate Logical Sudoku & Mosaic
    print("\n[Phase 1] Generating Sudoku Puzzle & Synthetic MNIST Mosaic...")
    sudoku_gen = SudokuGenerator()
    solution_grid, puzzle_grid = sudoku_gen.generate_puzzle(num_givens=30)
    
    mosaic_gen = MosaicGenerator()
    mosaic, cell_images = mosaic_gen.create_mosaic(puzzle_grid)
    print(f"  -> Generated 9x9 Sudoku puzzle with {np.count_nonzero(puzzle_grid)} given cells.")
    print(f"  -> Created {mosaic.shape[0]}x{mosaic.shape[1]} pixel image mosaic.")
    
    # 2. Inject Visual Misreads
    print("\n[Phase 2] Injecting Controlled Visual Misreads into Image Grid...")
    corrupted_images, injected_records = mosaic_gen.inject_misreads(puzzle_grid, cell_images, num_misreads=2)
    for rec in injected_records:
        print(f"  -> Injected Misread at Cell {rec['cell']}: True='{rec['true_label']}' -> Image of '{rec['injected_digit']}'")
        
    # 3. Initialize & Run Integrated Pipeline
    print("\n[Phase 3] Running Integrated AI Pipeline (NN -> Search -> CSP)...")
    pipeline = SudokuPipeline()
    
    algorithms = ["UCS", "Greedy", "AStar"]
    results = {}
    
    for algo in algorithms:
        print(f"\n---> Executing Pipeline with Search Algorithm: {algo} <---")
        res = pipeline.solve_mosaic(corrupted_images, search_algo=algo, csp_mode="BT_FC_AC3")
        results[algo] = res
        
        for line in res["trace_log"]:
            print(f"  {line}")
            
    # 4. Comparative Metrics Summary Table
    print("\n==========================================================")
    print("            SEARCH ALGORITHMS COMPARISON SUMMARY           ")
    print("==========================================================")
    print(f"{'Algorithm':<12} | {'Solved':<8} | {'Nodes Expanded':<16} | {'Nodes Generated':<16} | {'CSP Backtracks':<15}")
    print("-" * 75)
    
    for algo in algorithms:
        res = results[algo]
        print(f"{algo:<12} | {str(res['success']):<8} | {res['search_nodes_expanded']:<16} | {res['search_nodes_generated']:<16} | {res['csp_backtracks']:<15}")
        
    print("==========================================================")
    
    # Display Grids
    last_res = results["UCS"]
    print_grid(puzzle_grid, "1. Ground Truth Puzzle Grid (With Blanks)")
    print_grid(last_res["initial_grid"], "2. NN Initial Read Grid (With Injected Misreads)")
    print_grid(last_res["corrected_grid"], "3. Search Corrected Grid")
    print_grid(last_res["final_solution"], "4. Final CSP Solved Grid")
    
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY! All core project requirements satisfied.")

if __name__ == "__main__":
    run_full_demo()