"""
Data Download, Preprocessing, and Split Script for MNIST (Digits 1-9)
Fulfills P7 requirements: Excludes digit 0, normalizes pixels to [0,1],
splits train/val/test with a fixed seed, and outputs a Data Dictionary.
"""

import os
import json
import torch
import numpy as np
from torchvision import datasets, transforms

# Fixed random seed for reproducibility (University Requirement)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "./data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def process_and_save_data():
    print("--- Starting Step 2: MNIST Data Processing ---")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Download raw MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("Downloading raw MNIST dataset...")
    raw_train = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    raw_test = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    
    # Extract data arrays
    train_images = raw_train.data.numpy()  # (60000, 28, 28)
    train_labels = raw_train.targets.numpy()  # (60000,)
    
    test_images = raw_test.data.numpy()  # (10000, 28, 28)
    test_labels = raw_test.targets.numpy()  # (10000,)
    
    # 2. Filter out digit 0 (Digits 1-9 only as required by PDF)
    print("Filtering out digit 0 (keeping digits 1-9 only)...")
    train_mask = train_labels != 0
    test_mask = test_labels != 0
    
    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask]
    
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]
    
    # 3. Normalize pixel values from [0, 255] to float32 [0.0, 1.0]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # 4. Perform Train / Validation split (50,000 Train / 10,000 Val target approx)
    print("Splitting train set into Train and Validation subsets...")
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    
    val_size = 10000
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]
    
    X_train, y_train = train_images[train_idx], train_labels[train_idx]
    X_val, y_val = train_images[val_idx], train_labels[val_idx]
    X_test, y_test = test_images, test_labels
    
    # 5. Save processed arrays locally
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "mnist_digits_1_9.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"Processed datasets successfully saved to {PROCESSED_DIR}/mnist_digits_1_9.npz")
    
    # 6. Create Data Dictionary (University Deliverable Requirement)
    data_dictionary = {
        "dataset_name": "MNIST Handwritten Digits (Subset 1-9)",
        "source": "Yann LeCun MNIST database",
        "random_seed": SEED,
        "preprocessing_steps": [
            "Downloaded standard MNIST dataset.",
            "Filtered out all samples with label 0.",
            "Normalized raw uint8 pixel values [0, 255] to float32 range [0.0, 1.0].",
            "Shuffled and partitioned training images into train and validation sets."
        ],
        "image_shape": [28, 28],
        "label_classes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "split_counts": {
            "X_train": X_train.shape[0],
            "X_val": X_val.shape[0],
            "X_test": X_test.shape[0]
        }
    }
    
    dict_path = os.path.join(DATA_DIR, "data_dictionary.json")
    with open(dict_path, "w") as f:
        json.dump(data_dictionary, f, indent=4)
        
    print(f"Data dictionary saved to {dict_path}")
    print("\n--- Summary ---")
    print(f"Train samples: {X_train.shape[0]} | Shape: {X_train.shape}")
    print(f"Val samples:   {X_val.shape[0]} | Shape: {X_val.shape}")
    print(f"Test samples:  {X_test.shape[0]} | Shape: {X_test.shape}")
    print("Step 2 completed successfully!")

if __name__ == "__main__":
    process_and_save_data()