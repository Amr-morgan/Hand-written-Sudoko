# P7 — Handwritten Sudoku Reader and Solver

An end-to-end artificial intelligence system combining Convolutional Neural Networks (CNN), State-Space Search Correction (UCS, Greedy, A*), and Constraint Satisfaction Problem (CSP) solvers to digitize and solve handwritten Sudoku grid mosaics.

---

## 📁 Repository Structure

```text
Handwritten_Sudoku_IS/
├── data/                  # Dataset generator, mosaic renderer, and blank detection
├── nn/                    # Logistic Regression baseline and PyTorch CNN architecture
├── search/                # UCS, Greedy Best-First, and A* Search correction algorithms
├── csp/                   # Backtracking, Forward Checking, and AC-3 solvers
├── integration/           # Pipeline connecting NN -> Search -> CSP
├── eval/                  # NN calibration metrics and experiment suites
├── tests/                 # Automated PyTest unit and edge-case suites
├── make_data.py           # Preprocessing script generating data_dictionary.json
├── run_demo.py            # Official end-to-end university demonstration entry point
├── requirements.txt       # Project dependencies
└── README.md              # Documentation and execution instructions

1. Environment Setup:

1.python -m venv venv
2.venv\Scripts\activate.bat
3.pip install -r requirements.txt

2. Generate Preprocessed Dataset & Dictionary:

python make_data.py

3. Train Neural Network Model:

python -m nn.train

4. Run Official Demonstration Script:

python run_demo.py

5. Run Automated Unit Tests:

pytest

6. Execute Benchmark Experiments:

python -m eval.eval_experiments

---

### 4. How to Run Experiments in VS Code Terminal (cmd)

Execute in module mode:

```cmd
python -m eval.eval_experiments


