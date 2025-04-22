import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# ============================================
# 1) DATA LOADING & PREPROCESSING
# ============================================
def load_dataset(csv_path="../../data/train.csv"):
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep relevant columns
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
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Monthly_Balance",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
        "Credit_Score"
    ]
    df = df[cols_to_keep].dropna(subset=cols_to_keep)

    # Map target classes to 0,1,2
    df["Credit_Score"] = df["Credit_Score"].replace({"Poor":0, "Standard":1, "Good":2})

    # Convert numeric columns safely
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

    cat_cols = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    df.dropna(inplace=True)

    feature_cols = numeric_cols + cat_cols
    target_col = "Credit_Score"

    X = df[feature_cols].values
    y = df[target_col].values
    return X, y, feature_cols


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# 2) TWO-HIDDEN-LAYER MLP
# ============================================
class TwoHiddenMLP(nn.Module):
    """
    Input -> Linear(h1) -> ReLU -> Linear(h2) -> ReLU -> Linear(num_classes)

    We want the dimension *before* final Linear to be 128, 
    so final layer is shape (num_classes, 128).
    We'll define hidden_dim1=128, hidden_dim2=128. 
    That means final layer is nn.Linear(128, 3).
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=128, num_classes=3, dropout=0.2):
        super(TwoHiddenMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        out = self.fc3(x)  # shape (batch_size, num_classes)
        return out

    def final_hidden(self, x):
        """
        A helper function that returns the final hidden-layer output 
        BEFORE the last linear layer (fc3).
        => shape (batch_size, hidden_dim2=128)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x  # shape (batch_size, hidden_dim2)


# ============================================
# 3) TRAINING & EVAL
# ============================================
def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def main():
    # 1) Load data
    X, y, feat_cols = load_dataset("../../data/train.csv")
    print("Data shape:", X.shape, "Class distribution:", Counter(y))

    # 2) Train/Val/Test
    from sklearn.model_selection import train_test_split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # 3) (Optional) SMOTE if imbalanced
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print("After SMOTE, train shape:", X_train_sm.shape, Counter(y_train_sm))

    # 4) Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_sm = scaler.fit_transform(X_train_sm)
    X_val_scl  = scaler.transform(X_val)
    X_test_scl = scaler.transform(X_test)

    # 5) Build datasets
    train_ds = SimpleDataset(X_train_sm, y_train_sm)
    val_ds   = SimpleDataset(X_val_scl,  y_val)
    test_ds  = SimpleDataset(X_test_scl, y_test)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # 6) MLP: two hidden layers => final hidden dimension=128
    input_dim = X_train_sm.shape[1]
    model = TwoHiddenMLP(input_dim, hidden_dim1=128, hidden_dim2=128, num_classes=3, dropout=0.2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Weighted cross-entropy for any residual imbalance
    class_counts = np.bincount(y_train_sm)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 7) Train
    epochs = 50
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # 8) Load best model & test accuracy
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

    # 9) Now we extract the final hidden representation => shape(N, 128), 
    #    then save it as X_test.csv so your C++ code can do final-layer inference.
    model.eval()
    X_test_final = []
    with torch.no_grad():
        # We'll run the entire test set in one go, or in batches. 
        # Let's do a quick approach for demonstration:
        for i in range(X_test_scl.shape[0]):
            x_row = torch.tensor(X_test_scl[i], dtype=torch.float32, device=device).unsqueeze(0)  # shape(1, input_dim)
            h = model.final_hidden(x_row)  # shape(1,128)
            h_np = h.cpu().numpy().flatten().tolist()
            X_test_final.append(h_np)
    X_test_final = np.array(X_test_final)  # shape(N,128)

    # The final layer is 'fc3' => shape(3,128)
    # We'll get weight & bias
    final_weight = model.fc3.weight.data.cpu().numpy()  # shape(3,128)
    final_bias   = model.fc3.bias.data.cpu().numpy()    # shape(3,)

    print("Final layer shape:", final_weight.shape, final_bias.shape)
    # final_weight => (num_classes=3, hidden_dim2=128)
    # final_bias => (3,)

    # 10) Save these to CSV => C++ can replicate z = final_weight * X_test_final + final_bias
    np.savetxt("../../data/results_mlp/weight_matrix.csv", final_weight, delimiter=",")   # shape(3,128)
    np.savetxt("../../data/results_mlp/bias_vector.csv", final_bias[None], delimiter=",") # shape(1,3)
    np.savetxt("../../data/results_mlp/X_test.csv", X_test_final, delimiter=",")   # shape(N,128)
    np.savetxt("../../data/results_mlp/y_test.csv", y_test, delimiter=",")

    print("Saved: weight.csv", final_weight.shape, "bias.csv", final_bias.shape, "X_test.csv", X_test_final.shape, "y_test.csv", y_test.shape)
    print("Now your C++ code expecting final_dim=128 is consistent.")


if __name__ == "__main__":
    main()