"""
Tkinter GUI Interface for P7 — Handwritten Sudoku Reader & Solver.
Connects your trained CNN, Search engine, and CSP solver to a visual interface.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
import numpy as np

# Ensure root project directory is in system path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.generator import SudokuGenerator, MosaicGenerator
from integration.pipeline import SudokuPipeline


class SudokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("P7: Handwritten Sudoku AI Reader & Solver")
        self.root.resizable(False, False)

        # Initialize the AI pipeline
        self.pipeline = SudokuPipeline()
        self.cells = {}
        self.puzzle = None
        self.cell_images = None

        self._build_ui()
        self._generate_new_puzzle()

    def _build_ui(self):
        # 1. Top Control Toolbar Frame
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, pady=10)

        tk.Button(
            control_frame,
            text="🎲 Generate Puzzle",
            font=("Helvetica", 10, "bold"),
            command=self._generate_new_puzzle,
            padx=10, pady=5
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            control_frame,
            text="⚠️ Inject Misread",
            font=("Helvetica", 10, "bold"),
            command=self._inject_misread,
            padx=10, pady=5
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="🚀 Run AI Pipeline",
            font=("Helvetica", 10, "bold"),
            bg="#2E7D32",
            fg="white",
            command=self._solve_puzzle,
            padx=10, pady=5
        ).pack(side=tk.LEFT, padx=10)

        # 2. Sudoku 9x9 Grid Layout with 3x3 Block Borders
        grid_frame = tk.Frame(self.root, bg="#212121", bd=3)
        grid_frame.pack(padx=15, pady=15)

        for r in range(9):
            for c in range(9):
                top_pad = 3 if (r % 3 == 0 and r != 0) else 1
                left_pad = 3 if (c % 3 == 0 and c != 0) else 1

                cell_frame = tk.Frame(grid_frame, bg="white")
                cell_frame.grid(
                    row=r, column=c,
                    padx=(left_pad, 1),
                    pady=(top_pad, 1)
                )

                lbl = tk.Label(
                    cell_frame,
                    text="",
                    width=3,
                    height=1,
                    font=("Consolas", 18, "bold"),
                    bg="white",
                    fg="#212121"
                )
                lbl.pack(padx=4, pady=4)
                self.cells[(r, c)] = lbl

        # 3. Status Bar at the Bottom (Fixed pady=4)
        self.status_lbl = tk.Label(
            self.root,
            text="System Initialized.",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Helvetica", 9),
            bg="#e0e0e0",
            pady=4
        )
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X)

    def _generate_new_puzzle(self):
        """Generates a new valid Sudoku puzzle and corresponding visual mosaic."""
        sudoku_gen = SudokuGenerator()
        _, self.puzzle = sudoku_gen.generate_puzzle(num_givens=30)

        mosaic_gen = MosaicGenerator()
        _, self.cell_images = mosaic_gen.create_mosaic(self.puzzle)

        self._update_grid_display(self.puzzle, highlight_corrections={})
        self.status_lbl.config(text="Generated fresh 9x9 Sudoku puzzle with 30 givens.")

    def _inject_misread(self):
        """Simulates visual noise by swapping cell digit images."""
        mosaic_gen = MosaicGenerator()
        self.cell_images, records = mosaic_gen.inject_misreads(
            self.puzzle, self.cell_images, num_misreads=1
        )
        if records:
            rec = records[0]
            r, c = rec['cell']
            self.cells[(r, c)].config(fg="#D32F2F")
            self.status_lbl.config(
                text=f"Injected visual misread at cell {rec['cell']}: True='{rec['true_label']}' -> Injected image='{rec['injected_digit']}'"
            )

    def _solve_puzzle(self):
        """Passes image patches to the NN -> Search -> CSP Pipeline."""
        self.status_lbl.config(text="Running AI Pipeline (NN Reader -> Search Engine -> CSP Solver)...")
        self.root.update_idletasks()

        res = self.pipeline.solve_mosaic(
            self.cell_images, search_algo="UCS", csp_mode="BT_FC_AC3"
        )

        if res["success"]:
            self._update_grid_display(
                res["final_solution"], highlight_corrections=res["corrections"]
            )
            self.status_lbl.config(
                text=f"Success! Corrected misreads: {len(res['corrections'])} | Search nodes expanded: {res['search_nodes_expanded']} | CSP backtracks: {res['csp_backtracks']}"
            )
        else:
            messagebox.showerror(
                "Pipeline Failure",
                "The search engine could not resolve grid conflicts within maximum correction limits."
            )
            self.status_lbl.config(text="Pipeline execution failed.")

    def _update_grid_display(self, grid, highlight_corrections):
        """Renders 9x9 digit values with distinct color coding."""
        for r in range(9):
            for c in range(9):
                val = grid[r][c]
                lbl = self.cells[(r, c)]

                if (r, c) in highlight_corrections:
                    corrected_val = highlight_corrections[(r, c)]
                    lbl.config(text=str(corrected_val), fg="#7B1FA2", bg="#F3E5F5")
                elif self.puzzle[r][c] != 0:
                    lbl.config(text=str(val), fg="#212121", bg="white")
                elif val != 0:
                    lbl.config(text=str(val), fg="#1976D2", bg="white")
                else:
                    lbl.config(text="", bg="white")


if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuGUI(root)
    root.mainloop()