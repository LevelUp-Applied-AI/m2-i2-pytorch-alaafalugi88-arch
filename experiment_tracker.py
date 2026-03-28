import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json


# ─── Model ─────────────────────────────────────────────

class HousingModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Metrics ───────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return mae, r2


# ─── Main ──────────────────────────────────────────────

def main():
    df = pd.read_csv("data/housing.csv")

    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']

    X = df[feature_cols]
    y = df[['price_jod']]

    # 🔥 Feature scaling
    X_scaled = (X - X.mean()) / X.std()

    # 🔥 Target scaling
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    # Split
    split = int(0.8 * len(df))
    X_train = X_scaled[:split]
    X_test = X_scaled[split:]
    y_train = y_scaled[:split]
    y_test = y_scaled[split:]

    # To tensor
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # 🔥 Experiments
    learning_rates = [0.0005, 0.001, 0.005]
    hidden_sizes = [16, 32, 64]
    epochs_list = [100, 200]

    results = []

    for lr in learning_rates:
        for hidden in hidden_sizes:
            for epochs in epochs_list:

                model = HousingModel(hidden_size=hidden)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # Train
                for epoch in range(epochs):
                    preds = model(X_train)
                    loss = criterion(preds, y_train)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Predict
                with torch.no_grad():
                    preds = model(X_test)

                # 🔥 رجّع القيم الأصلية
                preds = preds.numpy() * y_std.values + y_mean.values
                y_true = y_test.numpy() * y_std.values + y_mean.values

                mae, r2 = compute_metrics(y_true, preds)

                print(f"MAE={mae:.2f}, R2={r2:.2f}, lr={lr}, hidden={hidden}, epochs={epochs}")

                results.append({
                    "lr": lr,
                    "hidden": hidden,
                    "epochs": epochs,
                    "mae": float(mae),
                    "r2": float(r2)
                })

    # 🔥 Save results
    with open("experiments.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n📁 Saved experiments.json")


# ─── Run ───────────────────────────────────────────────

if __name__ == "__main__":
    main()