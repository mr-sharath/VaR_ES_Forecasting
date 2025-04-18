# scripts/eval_var.py
import torch
import numpy as np
import pandas as pd
from utils import QRMogLSTM, create_sequences

def evaluate_var():
    # Load data
    imfs = np.load('data/processed/BLK_imfs.npy')
    if imfs.shape[0] > imfs.shape[1]:
        imfs = imfs.T
    
    # Create sequences with same window size as training
    X, y = create_sequences(imfs, window_size=10)  # Match trained window size
    
    # Load model
    model = QRMogLSTM(
        input_dim=imfs.shape[-1],
        hidden_dim=64,  # Match trained hidden_dim
        mog_steps=5     # Match trained mog_steps
    )
    model.load_state_dict(torch.load("models/best_qrmoglstm.pt"))
    model.eval()
    
    # Predict
    with torch.no_grad():
        predictions = model(X)
    
    # Calculate violations
    violations = (y.numpy().flatten() < predictions.numpy().flatten()).astype(int)
    violation_rate = violations.mean()
    
    print(f"VaR Backtest Results:")
    print(f"Violation Rate: {violation_rate:.4f} (Expected: 0.05)")
    print(f"Number of Violations: {violations.sum()}/{len(violations)}")

if __name__ == "__main__":
    evaluate_var()