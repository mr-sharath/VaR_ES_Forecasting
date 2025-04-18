# scripts/utils.py
import torch
import numpy as np
import torch.nn as nn

# scripts/utils.py (updated)
class QRMogLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, mog_steps=5):
        super().__init__()
        self.mog_steps = mog_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize Mogrifier parameters
        self.Q = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, hidden_dim)) 
            for _ in range(mog_steps//2 + 1)
        ])
        self.R = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, input_dim)) 
            for _ in range(mog_steps//2)
        ])
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Ensure input is 3D: (batch_size, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        
        # Mogrification steps
        for i in range(self.mog_steps):
            if i % 2 == 0:
                # Modify hidden state
                h = torch.sigmoid(x @ self.Q[i//2]) * h
            else:
                # Modify input
                x = torch.sigmoid(h.transpose(0,1) @ self.R[i//2]) * x
                
        # LSTM processing
        out, _ = self.lstm(x, (h, c))
        return self.linear(out[:, -1, :])  # Return last timestep

def quantile_loss(y_pred, y_true, tau):
    err = y_true - y_pred
    return torch.mean(torch.max((tau-1)*err, tau*err))

# scripts/utils.py (updated)
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X = np.array(X)
    y = np.array(y)
    
    # Ensure 3D shape: (samples, timesteps, features)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
    return torch.FloatTensor(X), torch.FloatTensor(y)
