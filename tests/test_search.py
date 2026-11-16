"""
Unit and Edge-Case Tests for Search Correction Engine.
Tests correction cost logic, zero-correction grids, and algorithm correctness.
"""

import pytest
import numpy as np
from csp.solver import SudokuCSP
from search.solver import SudokuSearch


@pytest.fixture
def dummy_probabilities():
    probs = {}
    for r in range(9):
        probs[r] = {}
        for c in range(9):
            p = {d: 0.05 for d in range(1, 10)}
            p[1] = 0.60
            probs[r][c] = p
    return probs


def test_search_zero_cost_same_digit(dummy_probabilities):
    grid = np.zeros((9, 9), dtype=int)
    searcher = SudokuSearch(grid, dummy_probabilities, SudokuCSP)
    cost = searcher.compute_step_cost((0, 0), 1, 1)
    assert cost == 0.0


def test_search_non_negative_cost(dummy_probabilities):
    grid = np.zeros((9, 9), dtype=int)
    searcher = SudokuSearch(grid, dummy_probabilities, SudokuCSP)
    # Changing from digit with high prob (1) to low prob (2) must produce positive cost
    cost = searcher.compute_step_cost((0, 0), 1, 2)
    assert cost > 0.0


def test_search_no_corrections_needed(dummy_probabilities):
    valid_puzzle = [
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
    
    for algo in ["UCS", "Greedy", "AStar"]:
        searcher = SudokuSearch(valid_puzzle, dummy_probabilities, SudokuCSP)
        solvable, grid, node, corrections = searcher.search(algorithm=algo)
        assert solvable is True
        assert len(corrections) == 0