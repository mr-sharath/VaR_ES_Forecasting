from vmdpy import VMD
import numpy as np
import pandas as pd

def decompose(signal, K=4):
    alpha = 2000       # Moderate bandwidth constraint
    tau = 0            # Noise-tolerance
    DC = 0             # No DC part imposed
    init = 1           # Initialize omegas uniformly
    tol = 1e-7
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    return u

# Example usage
train_returns = pd.read_csv('data/splits/BLK_train.csv')['Returns'].values
imfs = decompose(train_returns, K=4)
np.save('data/processed/BLK_imfs.npy', imfs)