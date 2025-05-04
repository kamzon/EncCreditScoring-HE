# binary_logistic_regression_export.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load dataset
file_path = "../../data/train.csv"
df = pd.read_csv(file_path)

# Filter only two classes: Poor (0) and Good (1)
df = df[df["Credit_Score"].isin(["Poor", "Good"])]
df["Credit_Score"] = df["Credit_Score"].replace({"Poor": 0, "Good": 1})

# Columns to use
cols = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month", "Monthly_Balance"
]
df = df[cols + ["Credit_Score"]].dropna()

# Convert numeric strings to float
def to_float_safe(x):
    try:
        return float(str(x).replace(",", ""))
    except:
        return np.nan

for col in cols:
    df[col] = df[col].apply(to_float_safe)

df = df.dropna()

# Prepare features and labels
X = df[cols].values
y = df["Credit_Score"].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Logistic Regression
class BinaryLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # No sigmoid

model = BinaryLogisticRegression(input_dim=X.shape[1])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(X_test_t))
    preds_class = (preds > 0.5).float()
    acc = (preds_class == y_test_t).float().mean().item()
    print(f"Test Accuracy: {acc:.4f}")

# Export weights
w = model.linear.weight.data.numpy().flatten()
b = model.linear.bias.data.numpy().item()

np.savetxt("../../data/result_binary_logreg/weight_vector.csv", w[None], delimiter=",", fmt="%.18f" )
np.savetxt("../../data/result_binary_logreg/bias_scalar.csv", [b], delimiter=",", fmt="%.18f")
np.savetxt("../../data/result_binary_logreg/X_test.csv", X_test, delimiter=",", fmt="%.18f")
np.savetxt("../../data/result_binary_logreg/y_test.csv", y_test, delimiter=",", fmt="%.18f")


print("Exported model parameters and test set.")