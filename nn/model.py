"""
Convolutional Neural Network (CNN) Architecture for MNIST Digit Classification (1-9).
Outputs raw logits and log-softmax probabilities required for search correction costs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SudokuDigitCNN(nn.Module):
    def __init__(self):
        super(SudokuDigitCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                           # 28x28 -> 14x14
        self.dropout1 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 9)  # 9 output classes (Digits 1 to 9)

    def forward(self, x):
        """
        Forward pass. Accepts tensor of shape (N, 1, 28, 28).
        Returns raw class logits of shape (N, 9).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

    def get_probabilities(self, x):
        """
        Returns softmax probability distribution over digits 1-9 for input images.
        Output dictionary maps digit (1-9) to probability float.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        # Map indices 0-8 to Sudoku digits 1-9
        return {digit: float(probs[digit - 1]) for digit in range(1, 10)}