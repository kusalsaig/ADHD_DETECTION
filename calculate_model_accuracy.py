import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Paths
BASE_DIR = r"d:\Mendeley-Sound-DS"
MODELS_DIR = os.path.join(BASE_DIR, "Models")
CSV_PATH = r"d:\Mendeley-Sound-DS\Web-Flask\Mendeley-Sound-DS-Flask\src\static\plots\feature_matrix.csv"

# Model Paths
SVM_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
GCN_PATH = os.path.join(MODELS_DIR, "Mendeley-Sound-DS-EEG-ADHD-Custom-GCN-Full.pt")
RF_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define Classes for GCN (Must match training definition) ---
class ResidualGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn  = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, A, X):
        H = torch.matmul(A, X)
        H = self.lin(H)
        H = self.bn(H)
        H = F.relu(H + self.res(X))
        return self.drop(H)

class CustomGCN(nn.Module):
    def __init__(self, in_dim=5, hid_dim=128, out_dim=2, n_nodes=5):
        super().__init__()
        self.A = nn.Parameter(torch.ones(n_nodes, n_nodes) / n_nodes)
        self.layer1 = ResidualGCNLayer(in_dim, hid_dim)
        self.layer2 = ResidualGCNLayer(hid_dim, hid_dim)
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, out_dim)
        )

    def forward(self, _, X):
        A_soft = torch.softmax(self.A, dim=1)
        H = self.layer1(A_soft, X)
        H = self.layer2(A_soft, H)
        if H.ndim == 3:
            return self.fc(H.mean(dim=1))
        else:
            return self.fc(H.mean(dim=0))

# --- Main Logic ---
def main():
    # --- Capture Output to File ---
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"Loading data from {CSV_PATH}...\n")
        
        if not os.path.exists(CSV_PATH):
            f.write(f"Error: Feature matrix not found at {CSV_PATH}\n")
            return

        df = pd.read_csv(CSV_PATH)

        # Extract features and labels
        # Features: delta, theta, alpha, beta, gamma
        X_raw = df[['delta', 'theta', 'alpha', 'beta', 'gamma']].values
        y_true = df['group'].apply(lambda x: 1 if x == 'ADHD' else 0).values

        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Class distribution: ADHD={sum(y_true)}, CONTROL={len(y_true)-sum(y_true)}\n")

        # Check Scaler for Abs vs Rel
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            scaler_mean_sum = np.sum(scaler.mean_)
            
            if scaler_mean_sum < 2.0: 
                f.write("Scaler statistics suggest RELATIVE bandpower training (Sum ~ 1.0).\n")
                total_power = X_raw.sum(axis=1, keepdims=True)
                total_power[total_power==0] = 1.0 
                X = X_raw / total_power
            else:
                f.write(f"Scaler statistics suggest ABSOLUTE bandpower training (Sum={scaler_mean_sum:.2f}).\n")
                X = X_raw
                
            f.write("Scaling features...\n")
            X_scaled = scaler.transform(X)
        else:
            f.write(f"Error: Scaler not found at {SCALER_PATH}\n")
            return

        # --- SVM Evaluation ---
        if os.path.exists(SVM_PATH):
            f.write("\n--- Evaluating SVM Model ---\n")
            svm_model = joblib.load(SVM_PATH)
            y_pred_svm = svm_model.predict(X_scaled)
            
            acc_svm = accuracy_score(y_true, y_pred_svm)
            f.write(f"SVM Accuracy: {acc_svm*100:.2f}%\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred_svm)) + "\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred_svm, target_names=['CONTROL', 'ADHD']) + "\n")
        else:
            f.write(f"SVM model not found at {SVM_PATH}\n")

        # --- Random Forest Evaluation ---
        if os.path.exists(RF_PATH):
            f.write("\n--- Evaluating Random Forest Model ---\n")
            rf_model = joblib.load(RF_PATH)
            y_pred_rf = rf_model.predict(X)
            
            acc_rf = accuracy_score(y_true, y_pred_rf)
            f.write(f"Random Forest Accuracy: {acc_rf*100:.2f}%\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred_rf)) + "\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred_rf, target_names=['CONTROL', 'ADHD']) + "\n")
        else:
            f.write(f"Random Forest model not found at {RF_PATH}\n")

        # --- GCN Evaluation ---
        if os.path.exists(GCN_PATH):
            f.write("\n--- Evaluating Custom GCN Model ---\n")
            try:
                gcn_model = torch.load(GCN_PATH, map_location=device, weights_only=False)
                gcn_model.eval()
                
                preds = []
                for i in range(len(X_scaled)):
                    valid_features = X_scaled[i]
                    X_sample_np = np.tile(valid_features, (5, 1)) # (5, 5)
                    X_t = torch.tensor(X_sample_np, dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        logits = gcn_model(None, X_t)
                        if logits.ndim > 1:
                            pred = torch.argmax(logits, dim=1).item()
                        else:
                            pred = torch.argmax(logits).item()
                        preds.append(pred)
                
                acc_gcn = accuracy_score(y_true, preds)
                f.write(f"GCN Accuracy: {acc_gcn*100:.2f}%\n")
                f.write("Confusion Matrix:\n")
                f.write(str(confusion_matrix(y_true, preds)) + "\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_true, preds, target_names=['CONTROL', 'ADHD']) + "\n")
                
            except Exception as e:
                f.write(f"Error loading/running GCN: {e}\n")
                import traceback
                traceback.print_exc(file=f)
        else:
            f.write(f"GCN model not found at {GCN_PATH}\n")

if __name__ == "__main__":
    main()

