"""
Integration 2 — PyTorch: Housing Price Prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():

    # 1. Load Data
    df = pd.read_csv('data/housing.csv')
    print("Data shape:", df.shape)

    # 2. Features & Target
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # 3. Standardize
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std

    # 4. Convert to Tensor
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    print("X shape:", X_tensor.shape)
    print("y shape:", y_tensor.shape)

    # 5. Model + Loss + Optimizer
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 6. Training Loop
    num_epochs = 100

    for epoch in range(num_epochs):
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # 7. Save Predictions
    with torch.no_grad():
        preds = model(X_tensor)

    results = pd.DataFrame({
        'actual': y_tensor.numpy().flatten(),
        'predicted': preds.numpy().flatten()
    })

    results.to_csv('predictions.csv', index=False)
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
