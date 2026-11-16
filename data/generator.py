"""
Sudoku Puzzle Generator and MNIST Image Mosaic Synthesizer
Generates valid Sudoku grids, creates image mosaics using MNIST digit images,
implements brightness-based blank detection, and injects controlled misreads.
"""

import os
import json
import random
import numpy as np

# Fixed random seed for generator reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class SudokuGenerator:
    """Handles logical Sudoku puzzle creation using backtracking."""
    
    @staticmethod
    def is_valid(grid, row, col, num):
        """Checks if placing 'num' at grid[row][col] violates Sudoku rules."""
        for i in range(9):
            if grid[row][i] == num or grid[i][col] == num:
                return False
                
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if grid[r][c] == num:
                    return False
        return True

    def solve_grid(self, grid):
        """Solves grid in-place using backtracking. Returns True if solvable."""
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if self.is_valid(grid, row, col, num):
                            grid[row][col] = num
                            if self.solve_grid(grid):
                                return True
                            grid[row][col] = 0
                    return False
        return True

    def generate_full_grid(self):
        """Generates a complete, valid 9x9 Sudoku grid."""
        grid = [[0 for _ in range(9)] for _ in range(9)]
        self.solve_grid(grid)
        return np.array(grid, dtype=int)

    def generate_puzzle(self, num_givens=30):
        """
        Creates a Sudoku puzzle with the requested number of givens (non-zero cells).
        Returns both the complete solution grid and the puzzle grid with blanks (0).
        """
        solution = self.generate_full_grid()
        puzzle = solution.copy()
        
        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)
        
        cells_to_remove = 81 - num_givens
        for r, c in cells[:cells_to_remove]:
            puzzle[r][c] = 0
            
        return solution, puzzle


class MosaicGenerator:
    """Converts logical Sudoku grids into MNIST image mosaics."""
    
    def __init__(self, npz_path="./data/processed/mnist_digits_1_9.npz"):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found at {npz_path}. Run make_data.py first!")
            
        data = np.load(npz_path)
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]
        
        # Organize MNIST test images by digit label (1-9)
        self.digit_pools = {d: self.X_test[self.y_test == d] for d in range(1, 10)}

    def sample_digit_image(self, digit):
        """Samples a random 28x28 image for a given digit class (1-9)."""
        idx = random.randint(0, len(self.digit_pools[digit]) - 1)
        return self.digit_pools[digit][idx]

    def create_mosaic(self, puzzle_grid):
        """
        Stitches 81 cell images (28x28) into a 9x9 Sudoku mosaic (252x252).
        Blank cells (0) are filled with solid black pixels (intensity 0.0).
        """
        mosaic = np.zeros((9 * 28, 9 * 28), dtype=np.float32)
        cell_images = np.zeros((9, 9, 28, 28), dtype=np.float32)
        
        for r in range(9):
            for c in range(9):
                val = puzzle_grid[r][c]
                if val == 0:
                    img = np.zeros((28, 28), dtype=np.float32)
                else:
                    img = self.sample_digit_image(val)
                    
                cell_images[r][c] = img
                mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img
                
        return mosaic, cell_images

    @staticmethod
    def detect_blank_by_brightness(cell_img, threshold=0.02):
        """
        University Requirement: Detect blank cells using a deterministic 
        brightness/pixel-density rule, NOT the neural network.
        Returns True if the cell is blank.
        """
        mean_brightness = np.mean(cell_img)
        return mean_brightness < threshold

    def inject_misreads(self, puzzle_grid, cell_images, num_misreads=2, confuse_map=None):
        """
        Injects visual misreads by swapping selected digit images with 
        visually ambiguous or hard digit images according to a documented rule.
        Default confuse_map swaps common confusing pairs (e.g., 7 <-> 1, 4 <-> 9).
        """
        if confuse_map is None:
            confuse_map = {7: 1, 1: 7, 4: 9, 9: 4, 3: 8, 8: 3, 5: 6, 6: 5}
            
        corrupted_grid = puzzle_grid.copy()
        injected_records = []
        
        given_cells = [(r, c) for r in range(9) for c in range(9) if puzzle_grid[r][c] != 0]
        random.shuffle(given_cells)
        
        injected_count = 0
        for r, c in given_cells:
            if injected_count >= num_misreads:
                break
            true_val = puzzle_grid[r][c]
            if true_val in confuse_map:
                wrong_val = confuse_map[true_val]
                confusing_img = self.sample_digit_image(wrong_val)
                cell_images[r][c] = confusing_img
                
                # Document the injection rule record
                injected_records.append({
                    "cell": (r, c),
                    "true_label": int(true_val),
                    "injected_digit": int(wrong_val),
                    "rule": f"Replaced digit {true_val} image with digit {wrong_val} image from MNIST pool."
                })
                injected_count += 1
                
        return cell_images, injected_records


def test_generator():
    print("--- Testing Step 3: Sudoku & Mosaic Generator ---")
    
    # 1. Generate Logical Puzzle
    sudoku = SudokuGenerator()
    solution, puzzle = sudoku.generate_puzzle(num_givens=30)
    print(f"[OK] Generated Sudoku Solution and Puzzle with {np.count_nonzero(puzzle)} givens.")
    
    # 2. Generate Mosaic
    mosaic_gen = MosaicGenerator()
    mosaic, cell_imgs = mosaic_gen.create_mosaic(puzzle)
    print(f"[OK] Created Image Mosaic of shape: {mosaic.shape} (252x252 pixels).")
    
    # 3. Test Brightness Detection Rule
    blank_count = 0
    non_blank_count = 0
    for r in range(9):
        for c in range(9):
            is_blank = MosaicGenerator.detect_blank_by_brightness(cell_imgs[r][c])
            if is_blank:
                blank_count += 1
            else:
                non_blank_count += 1
                
    print(f"[OK] Brightness Rule Detection: {blank_count} blanks, {non_blank_count} givens.")
    assert non_blank_count == 30, f"Expected 30 non-blank cells, detected {non_blank_count}"
    
    # 4. Inject Misreads
    cell_imgs_corrupted, records = mosaic_gen.inject_misreads(puzzle, cell_imgs, num_misreads=2)
    print(f"[OK] Injected {len(records)} misreads into the image grid:")
    for rec in records:
        print(f"     Cell {rec['cell']}: True={rec['true_label']} -> Injected Image of {rec['injected_digit']}")
        
    print("\n🎉 Step 3 completed successfully!")

if __name__ == "__main__":
    test_generator()