"""
Baseline Logistic Regression Classifier for MNIST (Digits 1-9)
Fulfills P7 project-specific requirement: serves as a benchmark baseline model
to evaluate against the Neural Network.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report

def train_and_eval_baseline(data_path="./data/processed/mnist_digits_1_9.npz", save_dir="./nn"):
    print("--- Starting Step 4: Logistic Regression Baseline ---")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file missing at {data_path}. Run make_data.py first!")
        
    data = np.load(data_path)
    X_train = data["X_train"]  # (N, 28, 28)
    y_train = data["y_train"]  # (N,)
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Flatten 28x28 image matrices into 784-dimensional vectors for scikit-learn
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Flattened dataset shape for Logistic Regression: {X_train_flat.shape}")
    print("Training Logistic Regression model (lbfgs solver)...")
    
    # Initialize and train Logistic Regression baseline model
    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    model.fit(X_train_flat, y_train)
    
    # Evaluate on Validation and Test sets
    val_preds = model.predict(X_val_flat)
    val_probs = model.predict_proba(X_val_flat)
    val_acc = accuracy_score(y_val, val_preds)
    val_loss = log_loss(y_val, val_probs)
    
    test_preds = model.predict(X_test_flat)
    test_probs = model.predict_proba(X_test_flat)
    test_acc = accuracy_score(y_test, test_preds)
    test_loss = log_loss(y_test, test_probs)
    
    print("\n--- Baseline Results ---")
    print(f"Validation Accuracy: {val_acc * 100:.2f}% | Log-Loss: {val_loss:.4f}")
    print(f"Test Accuracy:       {test_acc * 100:.2f}% | Log-Loss: {test_loss:.4f}")
    
    # Print sample classification report
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, digits=4))
    
    # Save model weights to disk
    model_path = os.path.join(save_dir, "baseline_logistic.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Baseline model saved to {model_path}")
    print("\n🎉 Step 4 completed successfully!")

if __name__ == "__main__":
    train_and_eval_baseline()