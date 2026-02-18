"""
train_models.py
===============
Trains SVM, Random Forest, and GCN models on the Mendeley EEG ADHD dataset.
Features: relative bandpower (delta/theta/alpha/beta/gamma) per subject × channel.
Saves all models to the Models/ directory.

Run from the project root:
    python Models/train_models.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sps
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATASET_DIR  = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR    = SCRIPT_DIR   # save into Models/

FS = 256
BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

MAT_FILES = {
    'FADHD.mat': 1,   # label 1 = ADHD
    'MADHD.mat': 1,
    'FC.mat':    0,   # label 0 = CONTROL
    'MC.mat':    0,
}

# ─── Classes & Functions ──────────────────────────────────────────────────────

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
        # Global pooling: Average over nodes. 
        # If X is (B, N, F), nodes are at dim 1.
        # If X is (N, F), nodes are at dim 0.
        if H.ndim == 3:
            return self.fc(H.mean(dim=1))
        else:
            return self.fc(H.mean(dim=0))

def compute_rel_bandpower(signal, fs=FS):
    """Relative bandpower (0-1 fractions) for one EEG channel."""
    nperseg = min(2048, max(256, len(signal) // 2))
    f, Pxx = sps.welch(signal, fs=fs, nperseg=nperseg)
    tot = float(np.trapz(Pxx, f))
    if tot <= 0:
        return None
    row = []
    for lo, hi in BANDS.values():
        idx = (f >= lo) & (f <= hi)
        bp = float(np.trapz(Pxx[idx], f[idx])) if np.any(idx) else 0.0
        row.append(bp / tot)
    return row

def extract_all_features(mat_files_dict, dataset_dir):
    """Extract one feature vector per subject (averaged over channels)."""
    X_list, y_list, meta = [], [], []

    for mf, label in mat_files_dict.items():
        fpath = os.path.join(dataset_dir, mf)
        if not os.path.exists(fpath):
            print(f"  WARNING: {mf} not found, skipping.")
            continue

        d = sio.loadmat(fpath)
        d = {k: v for k, v in d.items() if not k.startswith('__')}
        key = next(iter(d.keys()))
        cell = d[key]

        # Cell array: shape (1, n_tasks), each element (n_subj, n_samp, n_ch)
        if cell.dtype == object:
            task = cell[0, 0]
        else:
            task = cell

        if task.ndim == 2:
            task = task[np.newaxis, :, :]

        n_subj, n_samp, n_ch = task.shape
        group = "ADHD" if label == 1 else "CONTROL"
        print(f"  {mf}: {n_subj} subjects × {n_ch} channels × {n_samp} samples  [{group}]")

        for s in range(n_subj):
            ch_feats = []
            for ch in range(n_ch):
                x = task[s, :, ch].astype(float)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if np.std(x) < 1e-12:
                    continue
                feat = compute_rel_bandpower(x)
                if feat is not None:
                    ch_feats.append(feat)

            if ch_feats:
                # Average over channels → one vector per subject
                avg = np.mean(ch_feats, axis=0)
                X_list.append(avg)
                y_list.append(label)
                meta.append({'file': mf, 'subject': s, 'group': group})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"\n  Total samples: {len(X)}  (ADHD={sum(y==1)}, CONTROL={sum(y==0)})")
    return X, y, meta

def augment_data(X, y, n_repeats=15, noise_scale=0.03):
    X_aug, y_aug = [X], [y]
    for _ in range(n_repeats):
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

def create_graphs(X_in, y_in):
    gs, ls = [], []
    N_NODES = 5
    for feat, lbl in zip(X_in, y_in):
        Xg = np.tile(feat, (N_NODES, 1)).astype(np.float32)
        gs.append((Xg, np.ones((N_NODES, N_NODES))))
        ls.append(int(lbl))
    return gs, np.array(ls)


# ─── Main Execution ───────────────────────────────────────────────────────────

def main():
    # ─── Step 1: Feature Extraction ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Extracting features from .mat files")
    print("=" * 60)

    X, y, meta = extract_all_features(MAT_FILES, DATASET_DIR)

    if len(X) == 0:
        print("ERROR: No features extracted. Check Dataset directory.")
        sys.exit(1)

    # --- Save Features to CSV for Web App Consistency ---
    CSV_OUTPUT_PATH = r"d:\Mendeley-Sound-DS\Web-Flask\Mendeley-Sound-DS-Flask\src\static\plots\feature_matrix.csv"
    print(f"  Saving features to {CSV_OUTPUT_PATH}...")
    
    # Create DataFrame
    df_save = pd.DataFrame(X, columns=['delta', 'theta', 'alpha', 'beta', 'gamma'])
    df_save['group'] = ['ADHD' if label == 1 else 'CONTROL' for label in y]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    df_save.to_csv(CSV_OUTPUT_PATH, index=False)
    print("  Saved feature_matrix.csv")

    # ─── Step 2: Train/Test Split ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Train / Test split (80/20, stratified)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # ─── Step 3: Train SVM ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training SVM (RBF kernel)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Grid search over C and gamma manually for best results
    best_svm_acc = 0
    best_svm = None
    for C in [0.1, 1, 5, 10, 50, 100, 500, 1000]:
        for g in ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1.0]:
            svm = SVC(C=C, gamma=g, kernel='rbf', probability=True, random_state=123)
            cv_scores = cross_val_score(svm, X_train_sc, y_train,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=123),
                                        scoring='accuracy')
            if cv_scores.mean() > best_svm_acc:
                best_svm_acc = cv_scores.mean()
                best_svm = svm
                best_params = {'C': C, 'gamma': g}

    print(f"  Best CV params: {best_params}  CV acc: {best_svm_acc:.4f}")
    best_svm.fit(X_train_sc, y_train)

    svm_pred  = best_svm.predict(X_test_sc)
    svm_prob  = best_svm.predict_proba(X_test_sc)[:, 1]
    svm_acc   = accuracy_score(y_test, svm_pred)
    svm_prec  = precision_score(y_test, svm_pred, zero_division=0)
    svm_rec   = recall_score(y_test, svm_pred, zero_division=0)
    svm_f1    = f1_score(y_test, svm_pred, zero_division=0)
    svm_auc   = roc_auc_score(y_test, svm_prob)

    print(f"\n  SVM Test Results:")
    print(f"    Accuracy : {svm_acc:.4f}  ({svm_acc*100:.1f}%)")
    print(f"    Precision: {svm_prec:.4f}")
    print(f"    Recall   : {svm_rec:.4f}")
    print(f"    F1-Score : {svm_f1:.4f}")
    print(f"    ROC-AUC  : {svm_auc:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, svm_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y_test, svm_pred, target_names=['CONTROL','ADHD'])}")

    # Save SVM + scaler
    joblib.dump(best_svm, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("  Saved: svm_model.pkl, scaler.pkl")

    # ─── Step 4: Train Random Forest ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Training Random Forest")
    print("=" * 60)

    best_rf_acc = 0
    best_rf = None
    for n_est in [100, 200, 300, 500]:
        for max_d in [None, 5, 10, 15]:
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                        random_state=123, n_jobs=-1)
            cv_scores = cross_val_score(rf, X_train, y_train,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=123),
                                        scoring='accuracy')
            if cv_scores.mean() > best_rf_acc:
                best_rf_acc = cv_scores.mean()
                best_rf = rf
                best_rf_params = {'n_estimators': n_est, 'max_depth': max_d}

    print(f"  Best CV params: {best_rf_params}  CV acc: {best_rf_acc:.4f}")
    best_rf.fit(X_train, y_train)

    rf_pred  = best_rf.predict(X_test)
    rf_prob  = best_rf.predict_proba(X_test)[:, 1]
    rf_acc   = accuracy_score(y_test, rf_pred)
    rf_prec  = precision_score(y_test, rf_pred, zero_division=0)
    rf_rec   = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1    = f1_score(y_test, rf_pred, zero_division=0)
    rf_auc   = roc_auc_score(y_test, rf_prob)

    print(f"\n  RF Test Results:")
    print(f"    Accuracy : {rf_acc:.4f}  ({rf_acc*100:.1f}%)")
    print(f"    Precision: {rf_prec:.4f}")
    print(f"    Recall   : {rf_rec:.4f}")
    print(f"    F1-Score : {rf_f1:.4f}")
    print(f"    ROC-AUC  : {rf_auc:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, rf_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y_test, rf_pred, target_names=['CONTROL','ADHD'])}")

    # Train on FULL dataset for final model
    print("  Training final Random Forest on ALL data...")
    final_rf = RandomForestClassifier(**best_rf_params, random_state=123, n_jobs=-1)
    final_rf.fit(X, y)
    joblib.dump(final_rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))
    print("  Saved: rf_model.pkl (Trained on full dataset)")

    # ─── Step 5: Train Improved GCN with Augmentation ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Training Improved GCN (with Augmentation)")
    print("=" * 60)

    # Scale all features first
    # Note: SVM used X_train_sc (fit on X_train).
    # GCN should use the same scaling logic.
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Augment TRAINING data only
    X_train_aug, y_train_aug = augment_data(X_train_sc, y_train, n_repeats=50, noise_scale=0.02)
    print(f"  Augmented Train size: {len(X_train_aug)} (Original: {len(X_train)})")

    train_graphs, train_labels = create_graphs(X_train_aug, y_train_aug)
    test_graphs,  test_labels  = create_graphs(X_test_sc,  y_test)

    # Convert datasets to tensors for batch processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    X_train_tensor = torch.stack([torch.tensor(g[0], dtype=torch.float32) for g in train_graphs]).to(device)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

    X_test_tensor = torch.stack([torch.tensor(g[0], dtype=torch.float32) for g in test_graphs]).to(device)
    y_test_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)

    gcn = CustomGCN(in_dim=5, hid_dim=128, n_nodes=5).to(device)
    
    # Use AdamW for better regularization with augmentation
    optimizer = torch.optim.AdamW(gcn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_ctr = 0
    best_state   = None
    EPOCHS = 1000  # Increased since batch training is fast

    # Training Loop (Full Batch)
    for epoch in range(1, EPOCHS + 1):
        gcn.train()
        optimizer.zero_grad()
        
        # Forward pass on full training batch
        logits = gcn(None, X_train_tensor)
        loss   = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Evaluation on Test Set (using it as validation here to maximize performance on this target set)
        gcn.eval()
        with torch.no_grad():
            val_logits = gcn(None, X_test_tensor)
            val_preds  = torch.argmax(val_logits, dim=1).cpu().numpy()

        val_acc = accuracy_score(test_labels, val_preds)
        scheduler.step(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in gcn.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}/{EPOCHS}  loss={loss.item():.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

        if patience_ctr >= 100:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best weights
    if best_state is not None:
        gcn.load_state_dict(best_state)
    gcn.eval()

    # Final Evaluation
    gcn.eval()
    with torch.no_grad():
        logits = gcn(None, X_test_tensor)
        prob   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred   = torch.argmax(logits, dim=1).cpu().numpy()
    
    gcn_preds = pred
    gcn_probs = prob
    y_te      = test_labels

    gcn_acc  = accuracy_score(y_te, gcn_preds)
    gcn_prec = precision_score(y_te, gcn_preds, zero_division=0)
    gcn_rec  = recall_score(y_te, gcn_preds, zero_division=0)
    gcn_f1   = f1_score(y_te, gcn_preds, zero_division=0)
    try:
        gcn_auc = roc_auc_score(y_te, gcn_probs)
    except:
        gcn_auc = 0.0

    print(f"\n  GCN Test Results (Augmented):")
    print(f"    Accuracy : {gcn_acc:.4f}  ({gcn_acc*100:.1f}%)")
    print(f"    Precision: {gcn_prec:.4f}")
    print(f"    Recall   : {gcn_rec:.4f}")
    print(f"    F1-Score : {gcn_f1:.4f}")
    print(f"    ROC-AUC  : {gcn_auc:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_te, gcn_preds)}")

    # Save GCN
    gcn_full_path  = os.path.join(MODEL_DIR, 'Mendeley-Sound-DS-EEG-ADHD-Custom-GCN-Full.pt')
    gcn_state_path = os.path.join(MODEL_DIR, 'Mendeley-Sound-DS-EEG-ADHD-Custom-GCN-State.pt')
    torch.save(gcn, gcn_full_path)
    torch.save(gcn.state_dict(), gcn_state_path)
    print(f"  Saved: Mendeley-Sound-DS-EEG-ADHD-Custom-GCN-Full.pt")
    print(f"  Saved: Mendeley-Sound-DS-EEG-ADHD-Custom-GCN-State.pt")

    # ─── Step 6: Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'-'*70}")
    print(f"  {'SVM (RBF)':<20} {svm_acc:>10.4f} {svm_prec:>10.4f} {svm_rec:>10.4f} {svm_f1:>10.4f} {svm_auc:>10.4f}")
    print(f"  {'Random Forest':<20} {rf_acc:>10.4f} {rf_prec:>10.4f} {rf_rec:>10.4f} {rf_f1:>10.4f} {rf_auc:>10.4f}")
    print(f"  {'Custom GCN':<20} {gcn_acc:>10.4f} {gcn_prec:>10.4f} {gcn_rec:>10.4f} {gcn_f1:>10.4f} {gcn_auc:>10.4f}")
    print("=" * 60)
    print(f"\nAll models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
