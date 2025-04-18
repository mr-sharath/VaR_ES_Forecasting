# scripts/train_var.py
import torch
import optuna
import numpy as np
from utils import QRMogLSTM, quantile_loss, create_sequences

# scripts/train_var.py (updated)
def objective(trial):
    # Hyperparameters
    window_size = trial.suggest_int('window_size', 5, 20)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    mog_steps = trial.suggest_int('mog_steps', 3, 10)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    # Load and prepare data
    imfs = np.load('data/processed/BLK_imfs.npy')
    
    # Ensure proper shape: (n_samples, n_features)
    if imfs.ndim == 1:
        imfs = imfs.reshape(-1, 1)
    elif imfs.shape[0] < imfs.shape[1]:
        imfs = imfs.T
    
    # Create sequences with correct dimensions
    X, y = create_sequences(imfs, window_size)
    
    # Train/val split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Model setup
    model = QRMogLSTM(
        input_dim=imfs.shape[-1],
        hidden_dim=hidden_dim,
        mog_steps=mog_steps
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = quantile_loss(preds, y_train, tau=0.05)
        loss.backward()
        optimizer.step()
    
    # Validation
    with torch.no_grad():
        val_loss = quantile_loss(model(X_val), y_val, tau=0.05)
    
    # Save best model
    if trial.number == 0 or val_loss < study.best_value:
        torch.save(model.state_dict(), "models/best_qrmoglstm.pt")
    
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    print("Best params:", study.best_params)