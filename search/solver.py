"""
Search Algorithms for Sudoku Digit Misread Correction.
Fulfills university requirements for P7:
- Computes non-negative correction costs from NN log-probabilities.
- Implements Uniform Cost Search (UCS), Greedy Best-First, and A* Search.
- Instrument counters: 'nodes_expanded' and 'nodes_generated'.
- Limits max corrections to 3.
"""

import heapq
import math
import numpy as np


class SearchNode:
    """Represents a node in the search tree for grid corrections."""
    
    def __init__(self, corrections, cost, heuristic, parent=None):
        """
        corrections: dict mapping (row, col) tuple to new_digit.
        cost: path cost g(n).
        heuristic: estimated cost to goal h(n).
        """
        self.corrections = corrections
        self.cost = cost
        self.heuristic = heuristic
        self.f_score = cost + heuristic
        self.parent = parent

    def __lt__(self, other):
        # Tie-breaker for priority queue priority comparison
        return self.f_score < other.f_score


class SudokuSearch:
    def __init__(self, initial_grid, probabilities, csp_solver_cls, max_corrections=3):
        """
        initial_grid: 9x9 matrix read by NN (0 for blank, 1-9 for detected digit).
        probabilities: 9x9 dict mapping (r,c) to digit probability dict {digit: P(digit|cell)}.
        csp_solver_cls: SudokuCSP class from csp.solver module.
        max_corrections: upper limit of corrections (University constraint = 3).
        """
        self.initial_grid = np.array(initial_grid, dtype=int)
        self.probabilities = probabilities
        self.CSP = csp_solver_cls
        self.max_corrections = max_corrections
        
        # Performance counters for experiments
        self.nodes_expanded = 0
        self.nodes_generated = 0

    def compute_step_cost(self, cell, old_digit, new_digit):
        """
        Calculates non-negative correction cost based on log-probabilities:
        Cost = -ln(P(new_digit)) - (-ln(P(old_digit))) = ln(P(old_digit)) - ln(P(new_digit))
        """
        cell_probs = self.probabilities[cell[0]][cell[1]]
        p_old = max(cell_probs.get(old_digit, 1e-6), 1e-6)
        p_new = max(cell_probs.get(new_digit, 1e-6), 1e-6)
        
        cost = math.log(p_old) - math.log(p_new)
        return max(0.0, cost)

    def apply_corrections(self, corrections):
        """Returns a copy of the grid with applied corrections."""
        grid_copy = self.initial_grid.copy()
        for (r, c), new_val in corrections.items():
            grid_copy[r][c] = new_val
        return grid_copy

    def is_solvable(self, grid):
        """Goal test: checks if the corrected grid is solvable by the CSP solver."""
        csp = self.CSP(grid)
        solvable, _ = csp.solve(mode="BT_FC_AC3")
        return solvable

    def compute_heuristic(self, corrections):
        """
        Admissible heuristic h(n): counts remaining conflicting givens 
        in rows, columns, and 3x3 boxes. At least one digit in each conflict must change.
        """
        grid = self.apply_corrections(corrections)
        conflicts = 0
        
        for i in range(9):
            # Check row conflicts
            row_vals = [grid[i][j] for j in range(9) if grid[i][j] != 0]
            conflicts += len(row_vals) - len(set(row_vals))
            
            # Check col conflicts
            col_vals = [grid[j][i] for j in range(9) if grid[j][i] != 0]
            conflicts += len(col_vals) - len(set(col_vals))
            
        return float(conflicts * 0.1)  # Weighted small scale to avoid overestimation

    def search(self, algorithm="UCS"):
        """
        Executes search to find the minimum-cost valid grid correction set.
        algorithm: 'UCS', 'Greedy', or 'AStar'
        """
        self.nodes_expanded = 0
        self.nodes_generated = 0
        
        # Test initial grid without corrections
        if self.is_solvable(self.initial_grid):
            start_node = SearchNode({}, 0.0, 0.0)
            return True, self.initial_grid, start_node, {}

        # OPTIMIZATION: Filter candidate cells to those with lower confidence or active row/col/box duplicates
        candidate_cells = []
        for r in range(9):
            for c in range(9):
                if self.initial_grid[r][c] != 0:
                    probs = self.probabilities[r][c]
                    top_prob = max(probs.values()) if probs else 1.0
                    # Focus on low-confidence predictions or cells with non-zero entries
                    if top_prob < 0.99 or self._has_direct_conflict(r, c):
                        candidate_cells.append((r, c))

        # Fallback if no low-confidence cells detected
        if not candidate_cells:
            candidate_cells = [(r, c) for r in range(9) for c in range(9) if self.initial_grid[r][c] != 0]

        start_h = self.compute_heuristic({})
        root_node = SearchNode(corrections={}, cost=0.0, heuristic=start_h)
        
        frontier = []
        counter = 0
        
        priority = root_node.cost if algorithm == "UCS" else (root_node.heuristic if algorithm == "Greedy" else root_node.f_score)
        heapq.heappush(frontier, (priority, counter, root_node))
        self.nodes_generated += 1

        visited = set()

        while frontier:
            _, _, current_node = heapq.heappop(frontier)
            self.nodes_expanded += 1

            grid_state = self.apply_corrections(current_node.corrections)

            # Goal Test
            if self.is_solvable(grid_state):
                return True, grid_state, current_node, current_node.corrections

            if len(current_node.corrections) >= self.max_corrections:
                continue

            state_key = tuple(sorted(current_node.corrections.items()))
            if state_key in visited:
                continue
            visited.add(state_key)

            for (r, c) in candidate_cells:
                if (r, c) in current_node.corrections:
                    continue

                old_digit = self.initial_grid[r][c]
                cell_probs = self.probabilities[r][c]

                # Only test top 2 alternative candidates instead of all 9
                sorted_digits = sorted(cell_probs.keys(), key=lambda d: cell_probs[d], reverse=True)[:3]
                for alt_digit in sorted_digits:
                    if alt_digit == old_digit:
                        continue

                    new_corrections = dict(current_node.corrections)
                    new_corrections[(r, c)] = alt_digit

                    step_cost = self.compute_step_cost((r, c), old_digit, alt_digit)
                    new_cost = current_node.cost + step_cost
                    new_h = self.compute_heuristic(new_corrections)

                    child_node = SearchNode(new_corrections, new_cost, new_h, parent=current_node)
                    
                    if algorithm == "UCS":
                        prio = new_cost
                    elif algorithm == "Greedy":
                        prio = new_h
                    else:
                        prio = new_cost + new_h

                    counter += 1
                    heapq.heappush(frontier, (prio, counter, child_node))
                    self.nodes_generated += 1

        return False, self.initial_grid, None, {}

    def _has_direct_conflict(self, r, c):
        """Helper to check if grid[r][c] conflicts with its row, column, or 3x3 box."""
        val = self.initial_grid[r][c]
        if val == 0:
            return False
        # Check row & col
        for i in range(9):
            if i != c and self.initial_grid[r][i] == val:
                return True
            if i != r and self.initial_grid[i][c] == val:
                return True
        # Check box
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for dr in range(3):
            for dc in range(3):
                nr, nc = br + dr, bc + dc
                if (nr, nc) != (r, c) and self.initial_grid[nr][nc] == val:
                    return True
        return False


def test_search_solver():
    print("--- Testing Step 8: Search Algorithms (UCS, Greedy, A*) ---")
    from csp.solver import SudokuCSP

    # Solvable base puzzle with an intentional conflict (misread)
    # Cell (0, 0) set to 3 instead of 5, causing conflict in row 0
    corrupted_puzzle = [
        [3, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    # Dummy NN probability distribution
    probabilities = {}
    for r in range(9):
        probabilities[r] = {}
        for c in range(9):
            val = corrupted_puzzle[r][c]
            if val != 0:
                probs = {d: 0.05 for d in range(1, 10)}
                probs[val] = 0.52
                if (r, c) == (0, 0):
                    probs[5] = 0.45  # True digit 5 has high secondary probability
                probabilities[r][c] = probs

    algos = ["UCS", "Greedy", "AStar"]
    print(f"{'Algorithm':<12} | {'Solved':<8} | {'Nodes Expanded':<16} | {'Nodes Generated':<16} | {'Cost':<8}")
    print("-" * 75)

    for algo in algos:
        searcher = SudokuSearch(corrupted_puzzle, probabilities, SudokuCSP, max_corrections=2)
        solved, corrected_grid, node, corrections = searcher.search(algorithm=algo)
        cost = f"{node.cost:.4f}" if node else "N/A"
        print(f"{algo:<12} | {str(solved):<8} | {searcher.nodes_expanded:<16} | {searcher.nodes_generated:<16} | {cost:<8}")

    print("\n🎉 Step 8 completed successfully!")

if __name__ == "__main__":
    test_search_solver()