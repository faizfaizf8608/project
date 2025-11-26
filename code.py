import os
import math
import argparse
import io
import base64
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise ImportError("This script requires PyTorch. Install it with `pip install torch`") from e

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional SARIMA
try:
    import statsmodels.api as sm  # type: ignore
    SARIMA_AVAILABLE = True
except Exception:
    SARIMA_AVAILABLE = False

import matplotlib.pyplot as plt

# ---------------------------
# Synthetic dataset generator
# ---------------------------
def generate_synthetic_multivariate(n_steps: int = 1200, n_vars: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic multivariate time series DataFrame with trend, seasonality, interactions and noise.
    Columns: var_0 ... var_{n_vars-1}
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps)
    data = np.zeros((n_steps, n_vars), dtype=float)
    # base seasonalities and trends
    for i in range(n_vars):
        trend = 0.01 * (i+1) * t  # small linear trend differing by variable
        period = 24 + (i * 7)  # different periods
        season = (np.sin(2 * np.pi * t / period) + 0.5 * np.cos(2 * np.pi * t / (period/2)))
        noise = rng.normal(scale=0.5 + 0.1*i, size=n_steps)
        data[:, i] = trend + season + noise
    # add interactions: variable j depends on variable 0 and 1
    data[:, 2] += 0.3 * data[:, 0] + 0.2 * data[:, 1]
    data[:, 3] += 0.5 * np.roll(data[:, 1], 3)  # lagged influence
    data[:, 4] += 0.2 * data[:, 2] - 0.1 * np.sin(0.1 * t)
    cols = [f"var_{i}" for i in range(n_vars)]
    return pd.DataFrame(data, columns=cols)

# ---------------------------
# Dataset for PyTorch
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, input_window: int = 60, output_window: int = 10, scaler: StandardScaler = None):
        """
        data: (T, D) numpy array
        """
        self.x = data.astype(np.float32)
        self.input_window = input_window
        self.output_window = output_window
        self.n_samples = len(self.x) - (input_window + output_window) + 1
        self.scaler = scaler

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        start = idx
        xi = self.x[start:start+self.input_window]
        yi = self.x[start+self.input_window:start+self.input_window+self.output_window]
        return xi, yi

# ---------------------------
# Transformer model (PyTorch)
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, output_window: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_window = output_window

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim * output_window)
        )

    def forward(self, src):
        # src: (batch, seq_len, input_dim)
        x = self.input_projection(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        # transformer encoder (batch_first=True)
        x_enc = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # global pooling: take last timestep embedding (you can try mean pooling)
        x_last = x_enc[:, -1, :]  # (batch, d_model)
        out = self.decoder(x_last)  # (batch, input_dim * output_window)
        out = out.view(out.size(0), self.output_window, self.input_dim)
        return out, x_enc  # return encoder outputs to analyze attention via hooks if needed

# ---------------------------
# LSTM baseline
# ---------------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_window: int = 10, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.reg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim * output_window))
        self.output_window = output_window
        self.input_dim = input_dim

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden_dim)
        out = self.reg(last)
        out = out.view(out.size(0), self.output_window, self.input_dim)
        return out

# ---------------------------
# Training and evaluation utils
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)[0] if isinstance(model, TimeSeriesTransformer) else model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)[0] if isinstance(model, TimeSeriesTransformer) else model(xb)
            y_trues.append(yb.cpu().numpy())
            y_preds.append(preds.cpu().numpy())
    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)
    return y_trues, y_preds

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # flatten across time horizon and variables
    true_flat = y_true.reshape(-1, y_true.shape[-1])
    pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    mae = mean_absolute_error(true_flat, pred_flat)
    rmse = mean_squared_error(true_flat, pred_flat, squared=False)
    # MAPE with clipping
    denom = np.clip(np.abs(true_flat), 1e-6, None)
    mape = np.mean(np.abs((true_flat - pred_flat) / denom)) * 100.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}

# ---------------------------
# SARIMA baseline (optional)
# ---------------------------
def sarima_forecast(series: np.ndarray, steps: int = 10):
    if not SARIMA_AVAILABLE:
        raise RuntimeError("statsmodels not available in environment. Install with `pip install statsmodels` to run SARIMA baseline.")
    # Fit univariate SARIMA on each variable separately and produce forecasts
    forecasts = []
    for i in range(series.shape[1]):
        s = series[:, i]
        # Simple order selection could be improved; here we use ARIMA(1,1,1)(1,1,1,24) as an example
        model = sm.tsa.statespace.SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,24), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        f = res.forecast(steps=steps)
        forecasts.append(f.reshape(-1, 1))
    forecasts = np.hstack(forecasts)  # (steps, n_vars)
    return forecasts  # (steps, n_vars)

# ---------------------------
# Helpers: save model weights as base64
# ---------------------------
def save_model_weights_base64(model: nn.Module, path: str):
    """
    Serialize state_dict to a base64-encoded text file.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode('ascii')
    with open(path, 'w') as f:
        f.write(b64)
    print(f"Saved base64 weights to {path} (size: {os.path.getsize(path)} bytes)")

# ---------------------------
# Visualization helpers
# ---------------------------
def plot_attention_encoder_outputs(encoder_outputs: np.ndarray, title: str = "Encoder outputs (heatmap)", savepath: str = None):
    """
    encoder_outputs: (seq_len, d_model) or (batch, seq_len, d_model)
    We average across d_model if provided batch.
    """
    if encoder_outputs.ndim == 3:
        encoder_outputs = encoder_outputs[0]  # take first batch
    seq_len, d_model = encoder_outputs.shape
    heat = np.abs(encoder_outputs)  # magnitude as proxy for "importance"
    plt.figure(figsize=(10, 3))
    plt.imshow(heat.T, aspect='auto')
    plt.colorbar()
    plt.xlabel("Time step")
    plt.ylabel("Feature (d_model)")
    plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
        print(f"Saved plot to {savepath}")
    else:
        plt.show()
    plt.close()

# ---------------------------
# Putting it all together: main orchestration
# ---------------------------
def main(args):
    # Hyperparameters
    n_steps = 1200
    n_vars = 5
    input_window = 60
    output_window = 10
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate or load data
    df = generate_synthetic_multivariate(n_steps=n_steps, n_vars=n_vars, seed=123)
    data = df.values.astype(float)
    # train/val/test split
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(len(data) * train_frac)
    n_val = int(len(data) * val_frac)
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Scaling (fit on train)
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # Datasets and loaders
    train_ds = TimeSeriesDataset(train_scaled, input_window=input_window, output_window=output_window)
    val_ds = TimeSeriesDataset(val_scaled, input_window=input_window, output_window=output_window)
    test_ds = TimeSeriesDataset(test_scaled, input_window=input_window, output_window=output_window)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Models
    torch.manual_seed(0)
    model = TimeSeriesTransformer(input_dim=n_vars, d_model=64, nhead=4, num_layers=2, output_window=output_window).to(device)
    lstm_model = LSTMForecast(input_dim=n_vars, hidden_dim=64, num_layers=2, output_window=output_window).to(device)

    # Loss, optimizer
    loss_fn = nn.SmoothL1Loss()  # Huber-like (aka SmoothL1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_lstm = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

    # Optionally do a short demo training if requested
    epochs = 5 if args.demo else 0
    if epochs > 0:
        print(f"Starting demo training for {epochs} epochs on device {device}")
        for epoch in range(epochs):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
            lstm_tr_loss = train_one_epoch(lstm_model, train_loader, opt_lstm, loss_fn, device)
            print(f"Epoch {epoch+1}/{epochs} - Transformer loss: {tr_loss:.4f} - LSTM loss: {lstm_tr_loss:.4f}")

    # Evaluate
    print("Evaluating models on test set...")
    y_true_t, y_pred_t = evaluate_model(model, test_loader, device)
    y_true_l, y_pred_l = evaluate_model(lstm_model, test_loader, device)
    # invert scaling
    # rebuild full test arrays (note: datasets are sliding windows on scaled test segment)
    # We can inverse transform by reshaping
    y_true_t_inv = scaler.inverse_transform(y_true_t.reshape(-1, n_vars)).reshape(y_true_t.shape)
    y_pred_t_inv = scaler.inverse_transform(y_pred_t.reshape(-1, n_vars)).reshape(y_pred_t.shape)
    y_true_l_inv = scaler.inverse_transform(y_true_l.reshape(-1, n_vars)).reshape(y_true_l.shape)
    y_pred_l_inv = scaler.inverse_transform(y_pred_l.reshape(-1, n_vars)).reshape(y_pred_l.shape)

    metrics_trans = compute_metrics(y_true_t_inv, y_pred_t_inv)
    metrics_lstm = compute_metrics(y_true_l_inv, y_pred_l_inv)
    print("Transformer metrics:", metrics_trans)
    print("LSTM metrics:", metrics_lstm)

    # SARIMA baseline (one-step forecast repeated for simplicity)
    if SARIMA_AVAILABLE:
        print("Running SARIMA baseline on test split (this may take time)...")
        # Use last portion of training+val as series to forecast next steps for each sliding window start in test
        # For quick baseline, perform single-block forecast from end of val into test horizon
        combined = np.vstack([train_data, val_data, test_data])
        try:
            sarima_f = sarima_forecast(combined, steps=output_window)
            # replicate to shape (n_samples, output_window, n_vars)
            sarima_preds = np.tile(sarima_f.reshape(1, output_window, n_vars), (y_true_t.shape[0], 1, 1))
            sarima_preds_inv = sarima_preds  # already in original scale
            metrics_sarima = compute_metrics(y_true_t_inv, sarima_preds_inv[:y_true_t_inv.shape[0]])
            print("SARIMA metrics:", metrics_sarima)
        except Exception as e:
            print("SARIMA baseline failed:", e)
    else:
        print("statsmodels not available — skipping SARIMA baseline. To enable, install statsmodels.")

    # Save model weights as base64 text
    os.makedirs("weights", exist_ok=True)
    save_model_weights_base64(model, os.path.join("weights", "transformer_weights_base64.txt"))

    # Save attention proxy visualization (encoder outputs for first test sample)
    # Run a forward pass for a single batch to obtain encoder outputs
    for xb, yb in test_loader:
        xb = xb.to(device)
        model.eval()
        with torch.no_grad():
            out, enc = model(xb)
            enc_np = enc.cpu().numpy()
        plot_attention_encoder_outputs(enc_np, title="Encoder outputs magnitude (proxy for attention)", savepath="attention_plot.png")
        break

    # Save a small CSV of metrics and a brief comparative report
    report = {
        "transformer": metrics_trans,
        "lstm": metrics_lstm,
        "sarima_available": SARIMA_AVAILABLE
    }
    with open("reports_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(\"\"\"Run complete. Files generated:
- weights/transformer_weights_base64.txt
- attention_plot.png
- reports_summary.json
To reproduce full training, set args.demo=False and implement a full training schedule in this script (longer epochs, checkpointing, hyperparameter search).
\"\"\")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run short demo training and evaluation (quick).')
    args = parser.parse_args()
    main(args)
