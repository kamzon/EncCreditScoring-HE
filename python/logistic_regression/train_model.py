import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1) LOAD & PREPROCESS DATA
# -----------------------------
df = pd.read_csv("../../data/train.csv", low_memory=False)

cols_to_keep = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Payment_Behaviour",
    "Monthly_Balance",
    "Credit_Score"
]
df = df[cols_to_keep]
df = df.dropna()

# Map categorical target to integers
mapping = {"Poor": 0, "Standard": 1, "Good": 2}
df["Credit_Score"] = df["Credit_Score"].map(mapping)

# Convert numeric columns
def to_float_safe(x):
    try:
        return float(str(x).replace(",", ""))
    except:
        return np.nan

numeric_cols = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Monthly_Balance"
]
for col in numeric_cols:
    df[col] = df[col].apply(to_float_safe)

# Encode categorical columns
cat_cols = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

df = df.dropna()

feature_cols = numeric_cols + cat_cols
target_col = "Credit_Score"

X = df[feature_cols].values
y = df[target_col].values

# Scale numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Oversample to balance the classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# -----------------------------
# 2) TORCH DATASETS
# -----------------------------
class CreditDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Integer labels for CrossEntropyLoss
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

train_ds = CreditDataset(X_train, y_train)
test_ds = CreditDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# -----------------------------
# 3) LOGISTIC REGRESSION MODEL
# -----------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)  # Raw logits for CrossEntropyLoss

input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim=input_dim, num_classes=3)

# -----------------------------
# 4) TRAINING
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

# -----------------------------
# 5) TESTING
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# -----------------------------
# 6) EXPORT RESULTS
# -----------------------------
with torch.no_grad():
    weight_matrix = model.linear.weight.data.numpy()
    bias_vector = model.linear.bias.data.numpy()

print("Weight matrix shape:", weight_matrix.shape)
print("Bias vector shape:", bias_vector.shape)

np.savetxt("../../data/resluts_log_reg/weight_matrix.csv", weight_matrix, delimiter=",")
np.savetxt("../../data/resluts_log_reg/bias_vector.csv", bias_vector[None], delimiter=",")  # shape (1,3)
np.savetxt("../../data/resluts_log_reg/X_test.csv", X_test, delimiter=",")
np.savetxt("../../data/resluts_log_reg/y_test.csv", y_test, delimiter=",")