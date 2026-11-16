"""
Training Script for Sudoku Digit CNN.
Trains model on processed MNIST (1-9), tracks validation loss/accuracy,
and saves trained model weights to ./nn/mnist_cnn.pt.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from nn.model import SudokuDigitCNN

# Fixed seed for training reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_cnn(data_path="./data/processed/mnist_digits_1_9.npz", save_dir="./nn", epochs=5, batch_size=64, lr=0.001):
    print("--- Starting Step 5: Neural Network Training ---")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data missing at {data_path}. Run make_data.py first!")
        
    data = np.load(data_path)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32).unsqueeze(1)  # (N, 1, 28, 28)
    y_train = torch.tensor(data["y_train"] - 1, dtype=torch.long)              # Convert 1-9 labels to 0-8
    
    X_val = torch.tensor(data["X_val"], dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(data["y_val"] - 1, dtype=torch.long)
    
    X_test = torch.tensor(data["X_test"], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(data["y_test"] - 1, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SudokuDigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
    # Final Test Pass
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    test_acc = test_correct / test_total
    print(f"\nFinal Test Set Accuracy: {test_acc * 100:.2f}%")
    
    # Save Model Weights
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "mnist_cnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")
    print("\n🎉 Step 5 completed successfully!")

if __name__ == "__main__":
    train_cnn()