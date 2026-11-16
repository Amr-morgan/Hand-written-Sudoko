"""
Unit Tests for Data Generation, Brightness Blank Detection, and NN Outputs.
"""

import os
import pytest
import numpy as np
from data.generator import MosaicGenerator


def test_digit_zero_exclusion():
    data_path = "./data/processed/mnist_digits_1_9.npz"
    if os.path.exists(data_path):
        data = np.load(data_path)
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]
        
        assert 0 not in y_train
        assert 0 not in y_val
        assert 0 not in y_test


def test_brightness_blank_detection():
    # Solid black image (intensity 0.0)
    blank_patch = np.zeros((28, 28), dtype=np.float32)
    assert bool(MosaicGenerator.detect_blank_by_brightness(blank_patch)) is True
    
    # White patch with digits (intensity 1.0)
    non_blank_patch = np.ones((28, 28), dtype=np.float32)
    assert bool(MosaicGenerator.detect_blank_by_brightness(non_blank_patch)) is False