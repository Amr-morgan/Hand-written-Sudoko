"""
Neural Network Evaluation and Calibration Metrics Script
Fulfills P7 project-specific requirements: compares CNN vs Logistic Regression baseline,
calculates Accuracy, Precision, Recall, Confusion Matrix, and Expected Calibration Error (ECE).
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from nn.model import SudokuDigitCNN

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    Measures how well predicted confidence probabilities align with actual empirical accuracy.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return ece

def evaluate_models(data_path="./data/processed/mnist_digits_1_9.npz", nn_path="./nn/mnist_cnn.pt", baseline_path="./nn/baseline_logistic.pkl"):
    print("--- Starting Step 6: Neural Network Evaluation & Calibration ---")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file missing at {data_path}. Run make_data.py first!")
    if not os.path.exists(nn_path):
        raise FileNotFoundError(f"CNN model missing at {nn_path}. Run python -m nn.train first!")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline model missing at {baseline_path}. Run python nn/baseline.py first!")
        
    data = np.load(data_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_test_idx = y_test - 1  # Convert 1-9 to 0-8 for 0-indexing
    
    # 1. Evaluate Logistic Regression Baseline
    with open(baseline_path, "rb") as f:
        baseline_model = pickle.load(f)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    baseline_probs = baseline_model.predict_proba(X_test_flat)
    baseline_preds = np.argmax(baseline_probs, axis=1)
    
    baseline_acc = accuracy_score(y_test_idx, baseline_preds)
    baseline_prec, baseline_rec, _, _ = precision_recall_fscore_support(y_test_idx, baseline_preds, average='macro')
    baseline_ece = compute_ece(baseline_probs, y_test_idx)
    
    # 2. Evaluate CNN Neural Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = SudokuDigitCNN().to(device)
    cnn_model.load_state_dict(torch.load(nn_path, map_location=device))
    cnn_model.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = cnn_model(X_test_tensor)
        cnn_probs = F.softmax(logits, dim=1).cpu().numpy()
        
    cnn_preds = np.argmax(cnn_probs, axis=1)
    
    cnn_acc = accuracy_score(y_test_idx, cnn_preds)
    cnn_prec, cnn_rec, _, _ = precision_recall_fscore_support(y_test_idx, cnn_preds, average='macro')
    cnn_ece = compute_ece(cnn_probs, y_test_idx)
    cnn_cm = confusion_matrix(y_test_idx, cnn_preds)
    
    # Print Comparison Table
    print("\n=======================================================")
    print("        MODEL EVALUATION & CALIBRATION METRICS          ")
    print("=======================================================")
    print(f"{'Metric':<25} | {'Logistic Regression':<20} | {'Sudoku CNN':<15}")
    print("-" * 65)
    print(f"{'Accuracy':<25} | {baseline_acc*100:19.2f}% | {cnn_acc*100:14.2f}%")
    print(f"{'Precision (Macro)':<25} | {baseline_prec*100:19.2f}% | {cnn_prec*100:14.2f}%")
    print(f"{'Recall (Macro)':<25} | {baseline_rec*100:19.2f}% | {cnn_rec*100:14.2f}%")
    print(f"{'Calibration Error (ECE)':<25} | {baseline_ece:19.4f}  | {cnn_ece:14.4f}")
    print("=======================================================\n")
    
    print("CNN Confusion Matrix (Digits 1-9):")
    print(cnn_cm)
    print("\n🎉 Step 6 completed successfully!")

if __name__ == "__main__":
    evaluate_models()