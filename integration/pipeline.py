"""
End-to-End Integrated Pipeline for Handwritten Sudoku Reader and Solver.
Fulfills P7 integration requirements:
Connects NN Probability Reading -> AC-3 Conflict Detection -> Search Corrections -> CSP Grid Solving.
"""

import os
import sys
import torch
import numpy as np

# Add project root directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.generator import MosaicGenerator
from nn.model import SudokuDigitCNN
from csp.solver import SudokuCSP
from search.solver import SudokuSearch


class SudokuPipeline:
    def __init__(self, cnn_model_path="./nn/mnist_cnn.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = SudokuDigitCNN().to(self.device)
        
        if os.path.exists(cnn_model_path):
            self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
            self.cnn.eval()
        else:
            print(f"Warning: CNN weights not found at {cnn_model_path}. Please train model first!")

    def process_cell_images(self, cell_images):
        """
        Reads 9x9 cell images (28x28).
        Uses brightness rule for blank detection and CNN for digit classification.
        Returns:
          - initial_grid: 9x9 matrix of predicted digits (0 for blank).
          - probabilities: 9x9 dict of cell probability distributions.
        """
        initial_grid = np.zeros((9, 9), dtype=int)
        probabilities = {}
        
        for r in range(9):
            probabilities[r] = {}
            for c in range(9):
                cell_img = cell_images[r][c]
                
                # Step A: Brightness Rule for Blank Cells (University Requirement)
                if MosaicGenerator.detect_blank_by_brightness(cell_img):
                    initial_grid[r][c] = 0
                    probabilities[r][c] = {d: 0.0 for d in range(1, 10)}
                else:
                    # Step B: Pass image patch through CNN for digit probabilities
                    img_tensor = torch.tensor(cell_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                    probs_dict = self.cnn.get_probabilities(img_tensor)
                    
                    predicted_digit = max(probs_dict.keys(), key=lambda d: probs_dict[d])
                    initial_grid[r][c] = predicted_digit
                    probabilities[r][c] = probs_dict
                    
        return initial_grid, probabilities

    def solve_mosaic(self, cell_images, search_algo="UCS", csp_mode="BT_FC_AC3", max_corrections=3):
        """
        Executes full end-to-end integrated pipeline.
        Returns detailed summary dictionary with execution trace and solution.
        """
        trace_log = []
        trace_log.append("=== Starting Handwritten Sudoku Reading & Solving Pipeline ===")
        
        # 1. Image Classification & Blank Detection
        initial_grid, probabilities = self.process_cell_images(cell_images)
        trace_log.append(f"[NN Reader] Initial grid recognized with {np.count_nonzero(initial_grid)} non-blank givens.")
        
        # 2. Initial AC-3 Conflict Checking
        csp_initial = SudokuCSP(initial_grid)
        initial_solvable, initial_solved = csp_initial.solve(mode=csp_mode)
        
        if initial_solvable:
            trace_log.append("[CSP Engine] Initial read grid is clean and directly solvable (0 misreads detected).")
            return {
                "success": True,
                "initial_grid": initial_grid,
                "corrected_grid": initial_grid,
                "final_solution": initial_solved,
                "corrections": {},
                "search_nodes_expanded": 0,
                "search_nodes_generated": 0,
                "csp_assignments": csp_initial.assignments,
                "csp_backtracks": csp_initial.backtracks,
                "trace_log": trace_log
            }
            
        trace_log.append("[CSP Engine] Contradiction/Conflict detected in initial read grid! Triggering Search engine...")
        
        # 3. Search Engine Correction
        searcher = SudokuSearch(
            initial_grid=initial_grid,
            probabilities=probabilities,
            csp_solver_cls=SudokuCSP,
            max_corrections=max_corrections
        )
        
        search_solvable, corrected_grid, goal_node, corrections = searcher.search(algorithm=search_algo)
        
        trace_log.append(f"[Search Engine] Algorithm '{search_algo}' finished.")
        trace_log.append(f"  - Nodes Expanded:  {searcher.nodes_expanded}")
        trace_log.append(f"  - Nodes Generated: {searcher.nodes_generated}")
        
        if not search_solvable:
            trace_log.append("❌ Pipeline Failure: Search could not find a valid correction set within correction limits.")
            return {
                "success": False,
                "initial_grid": initial_grid,
                "corrected_grid": initial_grid,
                "final_solution": None,
                "corrections": {},
                "search_nodes_expanded": searcher.nodes_expanded,
                "search_nodes_generated": searcher.nodes_generated,
                "csp_assignments": 0,
                "csp_backtracks": 0,
                "trace_log": trace_log
            }
            
        trace_log.append(f"[Search Engine] Solution found with {len(corrections)} correction(s):")
        for cell, new_val in corrections.items():
            old_val = initial_grid[cell[0]][cell[1]]
            cost = searcher.compute_step_cost(cell, old_val, new_val)
            trace_log.append(f"  - Cell {cell}: Read '{old_val}' -> Corrected to '{new_val}' (Step Cost: {cost:.4f})")
            
        # 4. Final CSP Solving
        csp_final = SudokuCSP(corrected_grid)
        final_solvable, final_solution = csp_final.solve(mode=csp_mode)
        trace_log.append(f"[CSP Engine] Final grid solved successfully! (Assignments: {csp_final.assignments}, Backtracks: {csp_final.backtracks})")
        
        return {
            "success": True,
            "initial_grid": initial_grid,
            "corrected_grid": corrected_grid,
            "final_solution": final_solution,
            "corrections": corrections,
            "search_nodes_expanded": searcher.nodes_expanded,
            "search_nodes_generated": searcher.nodes_generated,
            "csp_assignments": csp_final.assignments,
            "csp_backtracks": csp_final.backtracks,
            "trace_log": trace_log
        }


def test_pipeline():
    print("--- Testing Step 9: System Integration Pipeline ---")
    
    # 1. Create puzzle and mosaic
    from data.generator import SudokuGenerator
    sudoku_gen = SudokuGenerator()
    solution, puzzle = sudoku_gen.generate_puzzle(num_givens=30)
    
    mosaic_gen = MosaicGenerator()
    mosaic, cell_imgs = mosaic_gen.create_mosaic(puzzle)
    
    # 2. Inject misreads
    cell_imgs_corrupted, injected_records = mosaic_gen.inject_misreads(puzzle, cell_imgs, num_misreads=2)
    print(f"Injected {len(injected_records)} test misread(s).")
    
    # 3. Run integrated pipeline
    pipeline = SudokuPipeline()
    results = pipeline.solve_mosaic(cell_imgs_corrupted, search_algo="UCS", csp_mode="BT_FC_AC3")
    
    print("\n" + "\n".join(results["trace_log"]))
    assert results["success"] == True, "Integrated Pipeline should successfully correct and solve the grid!"
    print("\n🎉 Step 9 completed successfully!")

if __name__ == "__main__":
    test_pipeline()