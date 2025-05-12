import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

df = pd.read_excel('10 Comment Unsorted.xlsx')
X = df.drop(['Label', 'Participants', 'Participant no'], axis=1).values
y = df['Label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

best_accuracy = 0.0
best_state = None
total_accuracy = 0.0
n_runs = 200

for random_state in range(1, n_runs + 1):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1/9, random_state=random_state
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = Net(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1000):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        predicted = (test_preds >= 0.5).float()
        accuracy = (predicted.eq(y_test_t).sum() / y_test_t.shape[0]).item()
        y_true_np = y_test_t.numpy().astype(int).ravel()
        y_pred_np = predicted.numpy().astype(int).ravel()
        f1 = f1_score(y_true_np, y_pred_np)

    print(f'Random State: {random_state:3d} | Test Accuracy: {accuracy:.3f} | F1 Score: {f1:.3f}')
    
    total_accuracy += accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_state = random_state

print(f'\nBest Random State: {best_state} with Test Accuracy: {best_accuracy:.3f}')
print(f'Average Test Accuracy over {n_runs} runs: {total_accuracy / n_runs:.3f}')
