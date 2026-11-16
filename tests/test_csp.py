"""
Unit and Edge-Case Tests for CSP Solver Module.
Tests standard solving, edge cases (invalid/unsolvable grids), and switchable modes.
"""

import pytest
from csp.solver import SudokuCSP


@pytest.fixture
def valid_puzzle():
    return [
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


@pytest.fixture
def invalid_puzzle():
    """Puzzle with an immediate direct row conflict (two 5s in row 0)."""
    return [
        [5, 5, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]


def test_csp_valid_puzzle_all_modes(valid_puzzle):
    for mode in ["BT", "BT_FC", "BT_FC_AC3"]:
        csp = SudokuCSP(valid_puzzle)
        solvable, solution = csp.solve(mode=mode)
        assert solvable is True
        assert solution is not None
        assert len(solution) == 9
        assert len(solution[0]) == 9


def test_csp_invalid_puzzle_rejection(invalid_puzzle):
    csp = SudokuCSP(invalid_puzzle)
    solvable, solution = csp.solve(mode="BT_FC_AC3")
    assert solvable is False
    assert solution is None


def test_csp_counters_increment(valid_puzzle):
    csp = SudokuCSP(valid_puzzle)
    csp.solve(mode="BT_FC")
    assert csp.assignments > 0