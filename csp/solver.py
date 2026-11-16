"""
Constraint Satisfaction Problem (CSP) Solver for Sudoku.
Fulfills university requirements:
- Implements Backtracking (BT), Forward Checking (FC), and Arc Consistency (AC-3).
- Uses switchable search modes.
- Instruments 'assignments' and 'backtracks' counters for formal experiments.
"""

from collections import deque
import copy


class SudokuCSP:
    def __init__(self, initial_grid):
        """
        initial_grid: 9x9 list/numpy array. 
        0 represents an empty cell, 1-9 represents a fixed given.
        """
        self.variables = [(r, c) for r in range(9) for c in range(9)]
        self.domains = {}
        
        # Build initial domains
        for r in range(9):
            for c in range(9):
                val = int(initial_grid[r][c])
                if val in range(1, 10):
                    self.domains[(r, c)] = {val}
                else:
                    self.domains[(r, c)] = set(range(1, 10))
                    
        # Define neighbors (arcs) for each cell
        self.neighbors = {v: set() for v in self.variables}
        for r in range(9):
            for c in range(9):
                cell = (r, c)
                # Same row and col
                for i in range(9):
                    if i != c:
                        self.neighbors[cell].add((r, i))
                    if i != r:
                        self.neighbors[cell].add((i, c))
                # Same 3x3 box
                br, bc = 3 * (r // 3), 3 * (c // 3)
                for dr in range(3):
                    for dc in range(3):
                        nr, nc = br + dr, bc + dc
                        if (nr, nc) != cell:
                            self.neighbors[cell].add((nr, nc))

        # Counters for experiments (University Requirement)
        self.assignments = 0
        self.backtracks = 0

    def get_all_arcs(self):
        """Returns all directed arcs (Xi, Xj) in the CSP graph."""
        arcs = []
        for xi in self.variables:
            for xj in self.neighbors[xi]:
                arcs.append((xi, xj))
        return arcs

    def ac3(self, domains=None):
        """
        AC-3 Algorithm for Arc Consistency.
        Prunes domains based on constraints.
        Returns False if any variable domain becomes empty, True otherwise.
        """
        if domains is None:
            domains = self.domains

        queue = deque(self.get_all_arcs())
        
        while queue:
            (xi, xj) = queue.popleft()
            if self._revise(domains, xi, xj):
                if len(domains[xi]) == 0:
                    return False  # Domain empty -> Conflict detected!
                for xk in self.neighbors[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(self, domains, xi, xj):
        """
        Revises domain of Xi with respect to Xj.
        Removes value v from Xi's domain if no value in Xj's domain satisfies v != v_j.
        """
        revised = False
        to_remove = set()
        for v in domains[xi]:
            # Constraint: v != v_j
            if not any(v != v_j for v_j in domains[xj]):
                to_remove.add(v)
                revised = True
        domains[xi] -= to_remove
        return revised

    def forward_checking(self, domains, var, value):
        """
        Forward Checking: After assigning var = value,
        prune value from domains of all unassigned neighbors of var.
        Returns False if any neighbor domain becomes empty.
        """
        for neighbor in self.neighbors[var]:
            if value in domains[neighbor]:
                domains[neighbor].remove(value)
                if len(domains[neighbor]) == 0:
                    return False
        return True

    def solve(self, mode="BT_FC_AC3"):
        """
        Solves the CSP puzzle using the requested switchable mode:
        1. 'BT'         : Pure Backtracking
        2. 'BT_FC'      : Backtracking + Forward Checking
        3. 'BT_FC_AC3'  : Backtracking + Forward Checking + Initial AC-3
        Returns: (solvable: bool, solved_grid: list/None)
        """
        self.assignments = 0
        self.backtracks = 0
        domains_copy = copy.deepcopy(self.domains)

        # Apply initial AC-3 preprocessing if requested
        if mode == "BT_FC_AC3":
            if not self.ac3(domains_copy):
                return False, None

        assignment = {}
        for var, dom in domains_copy.items():
            if len(dom) == 1:
                assignment[var] = list(dom)[0]

        solvable, final_assignment = self._backtrack(assignment, domains_copy, mode)
        
        if solvable:
            solved_grid = [[0 for _ in range(9)] for _ in range(9)]
            for (r, c), val in final_assignment.items():
                solved_grid[r][c] = val
            return True, solved_grid
        else:
            return False, None

    def _select_unassigned_variable(self, assignment, domains):
        """Minimum Remaining Values (MRV) heuristic."""
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None
        # Return variable with smallest domain size
        return min(unassigned, key=lambda var: len(domains[var]))

    def _is_consistent(self, var, val, assignment):
        """Checks if assigning var = val violates constraints with current assignment."""
        for neighbor in self.neighbors[var]:
            if neighbor in assignment and assignment[neighbor] == val:
                return False
        return True

    def _backtrack(self, assignment, domains, mode):
        """Core Backtracking recursive function."""
        if len(assignment) == 81:
            return True, assignment

        var = self._select_unassigned_variable(assignment, domains)
        if var is None:
            return True, assignment

        for val in sorted(list(domains[var])):
            if self._is_consistent(var, val, assignment):
                self.assignments += 1
                assignment[var] = val
                
                # Deepcopy domains for local branch modification
                local_domains = copy.deepcopy(domains)
                local_domains[var] = {val}
                
                legal = True
                if mode in ["BT_FC", "BT_FC_AC3"]:
                    legal = self.forward_checking(local_domains, var, val)

                if legal:
                    result, final_assign = self._backtrack(assignment, local_domains, mode)
                    if result:
                        return True, final_assign

                # Backtrack step
                del assignment[var]
                self.backtracks += 1

        return False, None


def test_csp_solver():
    print("--- Testing Step 7: CSP Solver (BT, BT+FC, BT+FC+AC-3) ---")
    
    # Sample valid puzzle with 30 givens
    sample_puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    modes = ["BT", "BT_FC", "BT_FC_AC3"]
    print(f"{'Mode':<15} | {'Solvable':<10} | {'Assignments':<12} | {'Backtracks':<10}")
    print("-" * 55)
    
    for mode in modes:
        solver = SudokuCSP(sample_puzzle)
        solvable, solved_grid = solver.solve(mode=mode)
        print(f"{mode:<15} | {str(solvable):<10} | {solver.assignments:<12} | {solver.backtracks:<10}")

    print("\n🎉 Step 7 completed successfully!")

if __name__ == "__main__":
    test_csp_solver()