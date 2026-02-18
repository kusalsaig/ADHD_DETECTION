from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
# NumPy 2.0 compatibility: trapz was removed, use trapezoid instead
try:
    np_trapz = np.trapz
except AttributeError:
    np_trapz = np.trapezoid

import pandas as pd
import scipy.io as sio
import scipy.signal as sps
from itertools import combinations
import io
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_theme(style='ticks', rc={'figure.dpi':110, 'savefig.dpi':200, 'axes.grid':True, 'grid.linestyle':'--'})
import joblib



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report
    )

# Torch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print('PyTorch not available; deep learning cells will be skipped or show skeletons.')



app = Flask(__name__)
app.secret_key = "Mendeley-Sound-DS-DB-%8uyg(&%"

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "Mendeley-Sound-DS-DB.db")



APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 2: go UP three levels
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, "../../../"))

# Step 3: Dataset folder
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
mat_files = ['FC.mat','MC.mat','FADHD.mat','MADHD.mat']
raw = {}
for mf in mat_files:
    fullp = os.path.join(DATASET_DIR, mf)
    if not os.path.exists(fullp):
        print(f'Warning: {mf} not found at {fullp}')
        continue
    print('Loading', mf)
    d = sio.loadmat(fullp)
    d = {k: v for k, v in d.items() if not k.startswith('__')}
    raw[mf] = d
    print(' -> Variables:', list(d.keys()))

# ---------------------------
# Helper: DB connection
# ---------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_user(username, password):
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # User likely exists
    conn.close()

# ---------------------------
# Initialize database (Run once)
# ---------------------------
def init_db():
    if not os.path.exists(DB_PATH):
        conn = get_db()
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()
        print("Database created.")

        # Create default admin
        create_user("admin", "admin123")
        print("Default admin created: admin / admin123")


@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username,password)).fetchone()
        conn.close()

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    out_dir = os.path.join("static", "dashboard")
    os.makedirs(out_dir, exist_ok=True)

    total_subjects = df_band["subject_idx"].nunique()
    adhd_count = df_band[df_band["group_type"] == "ADHD"]["subject_idx"].nunique()
    control_count = df_band[df_band["group_type"] == "CONTROL"]["subject_idx"].nunique()

    # ----------- GRAPH 1: Subjects Per Group -----------
    df_subj = df_band.groupby("group_type")["subject_idx"].nunique().reset_index()
    df_subj.rename(columns={"subject_idx": "subjects"}, inplace=True)

    plot1 = os.path.join(out_dir, "subjects_per_group.png")
    plt.figure(figsize=(6,4))
    sns.barplot(data=df_subj, x="group_type", y="subjects", palette="Blues")
    plt.title("Subjects Per Group")
    plt.xlabel("Group")
    plt.ylabel("Total Subjects")
    plt.tight_layout()
    plt.savefig(plot1)
    plt.close()

    # ----------- GRAPH 2: Overall Bandpower Heatmap -----------
    band_cols = ["delta_rel","theta_rel","alpha_rel","beta_rel","gamma_rel"]
    df_band_avg = df_band[band_cols].mean().to_frame(name="value")

    plot2 = os.path.join(out_dir, "bandpower_heatmap.png")
    plt.figure(figsize=(5,4))
    sns.heatmap(df_band_avg, annot=True, cmap="viridis")
    plt.title("Overall Average Bandpower")
    plt.tight_layout()
    plt.savefig(plot2)
    plt.close()

    return render_template(
        "dashboard.html",
        total_subjects=total_subjects,
        adhd_count=adhd_count,
        control_count=control_count,
        model_acc=91.25,
        plot1="dashboard/subjects_per_group.png",
        plot2="dashboard/bandpower_heatmap.png"
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))




@app.route("/cell-summary")
def cell_summary():
    data = {}
    for mf, d in raw.items():
        key = next(iter(d.keys()))
        data[mf] = d[key]

    summary_rows = []
    for mf, cell in data.items():
        group = mf.replace(".mat", "")
        n_tasks = cell.size

        for i in range(n_tasks):
            task = cell[0, i]
            if isinstance(task, np.ndarray):
                summary_rows.append({
                    "file": mf,
                    "group": group,
                    "task_id": i,
                    "subjects": task.shape[0],
                    "samples": task.shape[1],
                    "channels": task.shape[2]
                })

    return render_template("cell_summary.html", summary=summary_rows)


@app.route("/raw-samples")
def raw_samples():
    samples = {}

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        for i in range(cell.size):
            task = cell[0, i]
            if isinstance(task, np.ndarray) and task.shape[0] > 0:
                sig = task[0, :, 0]
                samples[f"{mf} - Task {i}"] = sig.tolist()
                break

    return render_template("raw_samples.html", samples=samples)



@app.route("/psd")
def psd():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_files = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        for i in range(cell.size):
            task = cell[0, i]
            if isinstance(task, np.ndarray) and task.shape[0] > 0:

                sig = task[0, :, 0]
                f, Pxx = sps.welch(sig, fs=256, nperseg=512)

                fname = f"psd_{mf}_task{i}.png".replace(".mat", "")
                fpath = os.path.join(out_dir, fname)

                plt.figure(figsize=(6,4))
                plt.semilogy(f, Pxx)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("PSD (Power/Hz)")
                plt.title(f"{mf} - Task {i}")
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()

                plot_files.append(fname)
                break

    return render_template("psd.html", images=plot_files)

FS = 256  # sampling rate

mat_files = ["FC.mat", "MC.mat", "FADHD.mat", "MADHD.mat"]

group_names = {
    "FC.mat": "Female_Control",
    "MC.mat": "Male_Control",
    "FADHD.mat": "Female_ADHD",
    "MADHD.mat": "Male_ADHD"
}

CHANNEL_NAMES = ["O1", "F3", "F4", "Cz", "Fz"]

BANDS = {
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta":  (13,30),
    "gamma": (30,45)
}

@app.route("/bandpower")
def bandpower():
    BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
             'beta': (13, 30), 'gamma': (30, 45)}

    records = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        group = group_names.get(mf, mf)

        for task_idx in range(cell.size):
            task = cell[0, task_idx]
            if not isinstance(task, np.ndarray):
                continue

            if task.ndim == 2:
                task = task[np.newaxis, :, :]

            n_subjects, n_samples, n_channels = task.shape
            ch_names = CHANNEL_NAMES[:n_channels]

            for subj in range(n_subjects):
                sig_all = task[subj].astype(float)
                if np.isnan(sig_all).any():
                    continue

                for ch in range(n_channels):
                    x = sig_all[:, ch]
                    f, Pxx = sps.welch(x, fs=FS, nperseg=min(2048, max(256, n_samples // 2)))
                    tot = np_trapz(Pxx, f)

                    row = {
                        'group': group,
                        'matfile': mf,
                        'task_idx': task_idx,
                        'subject_idx': subj,
                        'channel': ch_names[ch]
                    }

                    for b, (lo, hi) in BANDS.items():
                        idx = (f >= lo) & (f <= hi)
                        bp = np_trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0
                        row[f'{b}_abs'] = bp
                        row[f'{b}_rel'] = bp / tot if tot > 0 else 0.0

                    records.append(row)

            break

    df_band = pd.DataFrame.from_records(records)

    top10 = df_band.head(10).to_html(classes="table", index=False)
    sample10 = df_band.sample(10).to_html(classes="table", index=False)
    bottom10 = df_band.tail(10).to_html(classes="table", index=False)

    return render_template(
        "bandpower.html",
        top10=top10,
        sample10=sample10,
        bottom10=bottom10
    )




@app.route("/spectrogram")
def spectrogram():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_files = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        for i in range(cell.size):
            task = cell[0, i]
            if isinstance(task, np.ndarray) and task.shape[0] > 0:

                sig = task[0, :, 0]
                f, t, Sxx = sps.spectrogram(sig, fs=256, nperseg=256, noverlap=128)

                fname = f"spec_{mf}_task{i}.png".replace(".mat", "")
                fpath = os.path.join(out_dir, fname)

                plt.figure(figsize=(6,4))
                plt.pcolormesh(t, f, 10*np.log10(Sxx), shading="gouraud", cmap="viridis")
                plt.ylabel("Frequency (Hz)")
                plt.xlabel("Time (s)")
                plt.title(f"{mf} - Task {i}")
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()

                plot_files.append(fname)
                break

    return render_template("spectrogram.html", images=plot_files)



import pywt

@app.route("/cwt")
def cwt():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_files = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        for i in range(cell.size):
            task = cell[0, i]

            if isinstance(task, np.ndarray) and task.shape[0] > 0:
                sig = task[0, :, 0]

                widths = np.arange(1, 128)
                cwt_mat, freqs = pywt.cwt(sig, widths, 'cmor')

                fname = f"cwt_{mf}_task{i}.png".replace(".mat", "")
                fpath = os.path.join(out_dir, fname)

                plt.figure(figsize=(7, 5))
                plt.imshow(
                    np.abs(cwt_mat),
                    extent=[0, len(sig)/FS, freqs[-1], freqs[0]],
                    cmap="viridis",
                    aspect="auto",
                    origin="upper"
                )
                plt.colorbar()
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency")
                plt.title(f"{mf} - Task {i} (CWT Scalogram)")
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()

                plot_files.append(fname)
                break

    return render_template("cwt.html", images=plot_files)


from itertools import combinations

# Connectivity computation removed as per user request.

# @app.route("/pearson-correlation")
# def pearson_corr():
#     return render_template("index.html") # Redirect or show not available

# @app.route("/pearson-coherence")
# def coherence():
#     return render_template("index.html") # Redirect or show not available




@app.route("/subject-summary")
def subject_summary():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_files = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        for task_i in range(cell.size):
            task = cell[0, task_i]

            if isinstance(task, np.ndarray):
                n_subj, samples, channels = task.shape

                means = []
                stds = []

                for s in range(n_subj):
                    sig = task[s].mean(axis=1)
                    means.append(np.mean(sig))
                    stds.append(np.std(sig))

                fname = f"subject_summary_{mf}_task{task_i}.png".replace(".mat", "")
                fpath = os.path.join(out_dir, fname)

                plt.figure(figsize=(6,4))
                plt.plot(means, label="Mean Amplitude")
                plt.plot(stds, label="Std Amplitude")
                plt.xlabel("Subject Index")
                plt.ylabel("Value")
                plt.title(f"{mf} - Task {task_i}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()

                plot_files.append(fname)
                break

    return render_template("subject_summary.html", images=plot_files)

from scipy.stats import ttest_ind
from scipy.stats import ttest_ind

@app.route("/statistical-tests")
def stats():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    control_tbr = []
    adhd_tbr = []
    plot_files = []

    def compute_tbr(sig):
        f, Pxx = sps.welch(sig, fs=256, nperseg=512)
        theta = np_trapz(Pxx[(f >= 4) & (f <= 8)], f[(f >= 4) & (f <= 8)])
        beta = np_trapz(Pxx[(f >= 13) & (f <= 30)], f[(f >= 13) & (f <= 30)])
        return theta / beta if beta != 0 else 0

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        group = "ADHD" if "ADHD" in mf.upper() else "CONTROL"
        collected = []

        for task_i in range(cell.size):
            task = cell[0, task_i]

            if isinstance(task, np.ndarray):
                n_subj = task.shape[0]

                for s in range(n_subj):
                    sig = task[s][:, 0]
                    tbr = compute_tbr(sig)
                    collected.append(tbr)

                break

        if group == "CONTROL":
            control_tbr.extend(collected)
        else:
            adhd_tbr.extend(collected)

    t_stat, p_val = ttest_ind(control_tbr, adhd_tbr, equal_var=False)

    fname = "tbr_test.png"
    fpath = os.path.join(out_dir, fname)

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=[control_tbr, adhd_tbr])
    plt.xticks([0, 1], ["Control", "ADHD"])
    plt.ylabel("Theta/Beta Ratio (TBR)")
    plt.title(f"T-test p-value = {p_val:.5f}")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()

    plot_files.append(fname)

    return render_template("stats.html", images=plot_files, p_value=p_val)



@app.route("/feature-matrix")
def feature_matrix():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    rows = []

    def bandpower(sig):
        f, Pxx = sps.welch(sig, fs=256, nperseg=512)
        result = {}
        for b, (a1, a2) in bands.items():
            idx = (f >= a1) & (f <= a2)
            result[b] = float(np_trapz(Pxx[idx], f[idx]))
        return result

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        gender = "F" if mf.startswith("F") else "M"
        group = "ADHD" if "ADHD" in mf.upper() else "CONTROL"

        for task_i in range(cell.size):
            task = cell[0, task_i]

            if isinstance(task, np.ndarray):
                n_subj = task.shape[0]

                for s in range(n_subj):
                    sig = task[s][:, 0]
                    bp = bandpower(sig)

                    row = {
                        "subject_id": f"{mf}_S{s}",
                        "gender": gender,
                        "group": group,
                    }
                    row.update(bp)
                    rows.append(row)

                break

    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "feature_matrix.csv")
    df.to_csv(csv_path, index=False)

    top10 = df.head(10).to_html(classes="table", index=False)
    sample10 = df.sample(10).to_html(classes="table", index=False)
    bottom10 = df.tail(10).to_html(classes="table", index=False)

    return render_template(
        "feature_matrix.html",
        top10=top10,
        sample10=sample10,
        bottom10=bottom10,
        download="feature_matrix.csv"
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return self.fc(H.mean(dim=0))

def dl_prediction(delta, theta, alpha, beta, gamma):
    MODEL_PATH = os.path.join(MODEL_DIR, "Mendeley-Sound-DS-EEG-ADHD-Custom-GCN-Full.pt")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)

    # Scale inputs (matching training pipeline)
    X_raw = np.array([[delta, theta, alpha, beta, gamma]], dtype=np.float32)
    X_scaled = scaler.transform(X_raw)[0]

    # Build node feature vector (5 features)
    X = np.tile(X_scaled, (5, 1))  # (5,5)
    X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        # Passed A is ignored by the improved model
        logits = model(None, X_t)  # shape (2,)
        pred = torch.argmax(logits).item()

    result = "ADHD" if pred == 1 else "CONTROL"
    return result

def svm_prediction(delta, theta, alpha, beta, gamma):
    MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X = np.array([[delta, theta, alpha, beta, gamma]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    out = "ADHD" if pred == 1 else "CONTROL"
    return out

def rf_prediction(delta, theta, alpha, beta, gamma):
    MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
    model = joblib.load(MODEL_PATH)
    # RF uses unscaled features? Based on training checks, yes.
    # calculate_model_accuracy.py evaluated RF on X.
    # We pass raw features.
    X = np.array([[delta, theta, alpha, beta, gamma]])
    pred = model.predict(X)[0]
    return "ADHD" if pred == 1 else "CONTROL"

def extract_features_from_mat(mat_bytes):
    """Universal .mat EEG feature extractor â€” works with ANY .mat file format.

    Handles:
      - MATLAB v4 / v5 / v6 / v7.2  (via scipy.io)
      - MATLAB v7.3 HDF5 files       (via h5py)
      - Cell arrays  (object dtype)
      - Plain 1-D, 2-D, 3-D numeric arrays
      - Any variable name / multiple variables
      - Any orientation (samplesÃ—channels or channelsÃ—samples)
      - Nested structures / nested cell arrays

    Features are computed as RELATIVE bandpower (0-1 fractions), matching the
    scale used when the SVM / GCN models were trained.

    Returns:
        (delta, theta, alpha, beta, gamma,
         n_subjects, n_samples, n_channels, info_str, per_subj_list)
    Raises ValueError with a human-readable message on failure.
    """
    import io as _io

    buf = _io.BytesIO(mat_bytes)

    # â”€â”€ Step 1: Load the .mat file â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    d = None
    load_method = "scipy"
    try:
        buf.seek(0)
        raw_d = sio.loadmat(buf)
        d = {k: v for k, v in raw_d.items() if not k.startswith('__')}
    except Exception:
        # Fallback: MATLAB v7.3 (HDF5)
        try:
            import h5py
            buf.seek(0)
            d = {}
            with h5py.File(buf, 'r') as hf:
                def _visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            d[name.replace('/', '_')] = np.array(obj)
                        except Exception:
                            pass
                hf.visititems(_visit)
            load_method = "h5py (v7.3)"
        except Exception as e2:
            raise ValueError(
                f"Cannot read .mat file. Tried scipy.io and h5py. "
                f"Last error: {e2}"
            )

    if not d:
        raise ValueError("No readable data variables found in the .mat file.")

    # â”€â”€ Step 2: Recursively collect all numeric arrays â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def collect_arrays(obj, depth=0):
        found = []
        if depth > 8:
            return found
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                for elem in obj.flat:
                    found.extend(collect_arrays(elem, depth + 1))
            elif np.issubdtype(obj.dtype, np.number) and obj.size >= 64:
                found.append(obj.astype(float))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                found.extend(collect_arrays(item, depth + 1))
        return found

    candidates = []
    for val in d.values():
        candidates.extend(collect_arrays(val))

    if not candidates:
        raise ValueError(
            "No numeric arrays with enough data found in the .mat file. "
            "Make sure the file contains EEG signal data."
        )

    # â”€â”€ Step 3: Pick the best candidate array â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Prefer 3-D > 2-D > 1-D, then largest size
    def score(arr):
        ndim_score = {3: 300, 2: 200, 1: 100}.get(arr.ndim, 0)
        return ndim_score + min(arr.size, 10_000_000)

    candidates.sort(key=score, reverse=True)
    best = candidates[0]

    # â”€â”€ Step 4: Reshape to (n_subjects, n_samples, n_channels) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if best.ndim == 1:
        # Single channel signal
        eeg = best[np.newaxis, :, np.newaxis]           # (1, T, 1)

    elif best.ndim == 2:
        r, c = best.shape
        # Heuristic: longer axis = time (samples)
        if r >= c:
            eeg = best[np.newaxis, :, :]                # (1, T, C)
        else:
            eeg = best.T[np.newaxis, :, :]              # (1, T, C)

    elif best.ndim == 3:
        s, a, b = best.shape
        # Heuristic: axis with most elements = samples
        if a >= b:
            eeg = best                                   # (S, T, C)
        else:
            eeg = best.transpose(0, 2, 1)               # swap â†’ (S, T, C)

    else:
        # Flatten extra leading dims into subjects
        eeg = best.reshape(-1, best.shape[-2], best.shape[-1])

    n_subjects, n_samples, n_channels = eeg.shape

    # If orientation still looks wrong (very few samples), try transposing
    if n_samples < 64 and n_channels >= 64:
        eeg = eeg.transpose(0, 2, 1)
        n_subjects, n_samples, n_channels = eeg.shape

    if n_samples < 32:
        raise ValueError(
            f"Signal has only {n_samples} time-points after shape detection. "
            "Need at least 32 samples for frequency analysis. "
            "Please check your .mat file contains raw EEG time-series data."
        )

    # Cap channels to avoid runaway computation
    n_channels = min(n_channels, 64)

    # â”€â”€ Step 5: Compute relative bandpower (matches training pipeline) â”€â”€â”€â”€â”€â”€â”€â”€â”€
    nperseg = min(2048, max(256, n_samples // 2))

    all_rows = []

    for s in range(n_subjects):
        for ch in range(n_channels):
            x = eeg[s, :, ch]
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if np.all(x == 0) or np.std(x) < 1e-12:
                continue

            f_w, Pxx = sps.welch(x, fs=FS, nperseg=nperseg)
            tot = float(np_trapz(Pxx, f_w))
            if tot <= 0:
                continue

            row = []
            for _, (lo, hi) in BANDS.items():
                idx = (f_w >= lo) & (f_w <= hi)
                bp_val = float(np_trapz(Pxx[idx], f_w[idx])) if np.any(idx) else 0.0
                row.append(bp_val / tot)
            all_rows.append(row)

    if not all_rows:
        raise ValueError(
            "All extracted signals were flat/zero. "
            "The file may not contain valid EEG time-series data."
        )

    # Average across all subjects & channels â†’ single feature vector for prediction
    avg = np.mean(all_rows, axis=0)
    delta, theta, alpha, beta, gamma = avg

    # Per-subject averages (over channels) for display
    per_subj = []
    for s in range(n_subjects):
        chunk = all_rows[s * n_channels: (s + 1) * n_channels]
        if chunk:
            per_subj.append(np.mean(chunk, axis=0).tolist())

    # Best variable name for display
    best_key = next(iter(d.keys()), "unknown")
    info = (
        f"Variable: {best_key} | Format: {load_method} | "
        f"Subjects: {n_subjects} | Samples/subject: {n_samples} | Channels: {n_channels}"
    )
    return delta, theta, alpha, beta, gamma, n_subjects, n_samples, n_channels, info, per_subj


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        try:
            delta = float(request.form["delta"])
            theta = float(request.form["theta"])
            alpha = float(request.form["alpha"])
            beta  = float(request.form["beta"])
            gamma = float(request.form["gamma"])
        except:
            return render_template("predict.html", error="Invalid input values")

        svm_result = svm_prediction(delta, theta, alpha, beta, gamma)
        rf_result  = rf_prediction(delta, theta, alpha, beta, gamma)
        dl_result  = dl_prediction(delta, theta, alpha, beta, gamma)
        return render_template("predict.html", result=svm_result, rf_result=rf_result, ab_result=dl_result)

    return render_template("predict.html")


@app.route("/predict-mat", methods=["POST"])
def predict_mat():
    """Accept a .mat file upload and return ADHD / CONTROL prediction."""
    if 'mat_file' not in request.files:
        return render_template("predict.html", mat_error="No file was uploaded.")

    f = request.files['mat_file']
    if f.filename == '':
        return render_template("predict.html", mat_error="No file selected.")

    if not f.filename.lower().endswith('.mat'):
        return render_template("predict.html", mat_error="Please upload a valid .mat file.")

    try:
        mat_bytes = f.read()
        delta, theta, alpha, beta, gamma, n_subj, n_samp, n_ch, info, per_subj = \
            extract_features_from_mat(mat_bytes)
    except ValueError as e:
        return render_template("predict.html", mat_error=str(e))
    except Exception as e:
        return render_template("predict.html", mat_error=f"Unexpected error: {e}")

    try:
        svm_result = svm_prediction(delta, theta, alpha, beta, gamma)
    except Exception as e:
        svm_result = f"SVM error: {e}"

    try:
        dl_result = dl_prediction(delta, theta, alpha, beta, gamma)
    except Exception as e:
        dl_result = f"GCN error: {e}"

    # Display as percentages for readability (Ã—100), models receive raw 0-1 fractions
    features = {
        "Delta  (1-4 Hz)":  f"{delta  * 100:.2f}%",
        "Theta  (4-8 Hz)":  f"{theta  * 100:.2f}%",
        "Alpha  (8-13 Hz)": f"{alpha  * 100:.2f}%",
        "Beta  (13-30 Hz)": f"{beta   * 100:.2f}%",
        "Gamma (30-45 Hz)": f"{gamma  * 100:.2f}%",
    }

    filename = f.filename
    return render_template(
        "predict.html",
        mat_filename=filename,
        mat_info=info,
        mat_features=features,
        mat_svm_result=svm_result,
        mat_dl_result=dl_result
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CSV UPLOAD PREDICTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def extract_features_from_csv(csv_bytes):
    """Universal CSV EEG feature extractor.

    Handles two CSV formats automatically:

    FORMAT A â€” Bandpower columns already present:
        Columns contain names like delta/theta/alpha/beta/gamma (case-insensitive).
        Each row = one subject/sample. Features are read directly.
        Values can be absolute or relative; if they sum > 5 per row they are
        treated as percentages and divided by 100.

    FORMAT B â€” Raw EEG time-series:
        All columns are numeric (no band-name columns found).
        Each row = one time-point of EEG signal.
        Relative bandpower is computed from the full signal.

    Returns:
        (rows_data, csv_type, n_rows, col_info)
        rows_data : list of dicts with keys:
                    row_id, delta, theta, alpha, beta, gamma,
                    svm_result, dl_result
        csv_type  : "bandpower" or "timeseries"
        n_rows    : number of data rows processed
        col_info  : string describing detected columns
    Raises ValueError with a human-readable message.
    """
    import io as _io

    text = csv_bytes.decode('utf-8', errors='replace')
    # Try to read, skip blank / duplicate header rows (like SampleData.csv)
    try:
        df_raw = pd.read_csv(_io.StringIO(text))
    except Exception as e:
        raise ValueError(f"Cannot parse CSV: {e}")

    # Drop fully empty rows
    df_raw = df_raw.dropna(how='all').reset_index(drop=True)

    # Remove duplicate header rows (rows where all values equal column names)
    mask = df_raw.apply(
        lambda row: all(str(row[c]).strip().lower() == str(c).strip().lower()
                        for c in df_raw.columns),
        axis=1
    )
    df_raw = df_raw[~mask].reset_index(drop=True)

    if df_raw.empty:
        raise ValueError("CSV file is empty or contains only headers.")

    # â”€â”€ Detect FORMAT A: bandpower columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    BAND_ALIASES = {
        'delta': ['delta', 'delt', 'd'],
        'theta': ['theta', 'thet', 't'],
        'alpha': ['alpha', 'alph', 'a'],
        'beta':  ['beta',  'bet',  'b'],
        'gamma': ['gamma', 'gam',  'g'],
    }

    col_map = {}   # band_name -> actual column name
    cols_lower = {c: c.strip().lower() for c in df_raw.columns}

    for band, aliases in BAND_ALIASES.items():
        for col, col_l in cols_lower.items():
            if any(col_l == alias or col_l.startswith(alias) for alias in aliases):
                if band not in col_map:
                    col_map[band] = col

    has_all_bands = all(b in col_map for b in ['delta', 'theta', 'alpha', 'beta', 'gamma'])

    # â”€â”€ Detect a row-label / ID column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    id_col = None
    for c in df_raw.columns:
        cl = c.strip().lower()
        if cl in ('sample', 'id', 'subject', 'name', 'label', 'index', 'no', 'num'):
            id_col = c
            break

    rows_data = []

    if has_all_bands:
        # â”€â”€ FORMAT A: read bandpower values directly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        csv_type = "bandpower"
        col_info = (f"Detected bandpower columns: "
                    f"Delta={col_map['delta']}, Theta={col_map['theta']}, "
                    f"Alpha={col_map['alpha']}, Beta={col_map['beta']}, "
                    f"Gamma={col_map['gamma']}")

        for i, row in df_raw.iterrows():
            try:
                d_v = float(row[col_map['delta']])
                t_v = float(row[col_map['theta']])
                a_v = float(row[col_map['alpha']])
                b_v = float(row[col_map['beta']])
                g_v = float(row[col_map['gamma']])
            except (ValueError, TypeError):
                continue   # skip non-numeric rows

            # If values look like percentages (sum >> 1), normalise to 0-1
            total = d_v + t_v + a_v + b_v + g_v
            if total > 5.0:
                d_v /= total; t_v /= total; a_v /= total
                b_v /= total; g_v /= total

            row_id = str(row[id_col]) if id_col else f"Row {i+1}"

            try:
                svm_res = svm_prediction(d_v, t_v, a_v, b_v, g_v)
            except Exception as e:
                svm_res = f"Error: {e}"
            try:
                dl_res = dl_prediction(d_v, t_v, a_v, b_v, g_v)
            except Exception as e:
                dl_res = f"Error: {e}"

            rows_data.append({
                'row_id': row_id,
                'delta':  round(d_v * 100, 2),
                'theta':  round(t_v * 100, 2),
                'alpha':  round(a_v * 100, 2),
                'beta':   round(b_v * 100, 2),
                'gamma':  round(g_v * 100, 2),
                'svm':    svm_res,
                'gcn':    dl_res,
            })

    else:
        # â”€â”€ FORMAT B: treat all numeric columns as EEG time-series â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        csv_type = "timeseries"

        # Keep only numeric columns
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            # Try converting everything
            for c in df_raw.columns:
                try:
                    df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
                except Exception:
                    pass
            num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            raise ValueError(
                "No numeric columns found. For bandpower CSVs, name columns "
                "Delta, Theta, Alpha, Beta, Gamma. For time-series CSVs, "
                "all columns must be numeric."
            )

        col_info = (f"Detected time-series format: {len(num_cols)} numeric columns "
                    f"(treated as EEG channels), {len(df_raw)} time-points")

        # Each column = one EEG channel, rows = time-points
        signals = df_raw[num_cols].values.astype(float)   # (T, C)
        signals = np.nan_to_num(signals, nan=0.0)

        n_samples, n_channels = signals.shape
        nperseg = min(2048, max(64, n_samples // 2))

        band_vals = []
        for ch in range(min(n_channels, 64)):
            x = signals[:, ch]
            if np.std(x) < 1e-12:
                continue
            f_w, Pxx = sps.welch(x, fs=FS, nperseg=nperseg)
            tot = float(np_trapz(Pxx, f_w))
            if tot <= 0:
                continue
            row_bp = []
            for _, (lo, hi) in BANDS.items():
                idx = (f_w >= lo) & (f_w <= hi)
                bp_v = float(np_trapz(Pxx[idx], f_w[idx])) if np.any(idx) else 0.0
                row_bp.append(bp_v / tot)
            band_vals.append(row_bp)

        if not band_vals:
            raise ValueError("All signals in the CSV are flat/zero.")

        avg = np.mean(band_vals, axis=0)
        d_v, t_v, a_v, b_v, g_v = avg

        try:
            svm_res = svm_prediction(d_v, t_v, a_v, b_v, g_v)
        except Exception as e:
            svm_res = f"Error: {e}"
        try:
            dl_res = dl_prediction(d_v, t_v, a_v, b_v, g_v)
        except Exception as e:
            dl_res = f"Error: {e}"

        rows_data.append({
            'row_id': 'Full Signal',
            'delta':  round(d_v * 100, 2),
            'theta':  round(t_v * 100, 2),
            'alpha':  round(a_v * 100, 2),
            'beta':   round(b_v * 100, 2),
            'gamma':  round(g_v * 100, 2),
            'svm':    svm_res,
            'gcn':    dl_res,
        })

    if not rows_data:
        raise ValueError("No valid data rows could be processed from the CSV.")

    return rows_data, csv_type, len(rows_data), col_info


@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    """Accept a CSV file upload and return per-row ADHD / CONTROL predictions."""
    if 'csv_file' not in request.files:
        return render_template("predict.html", csv_error="No file was uploaded.")

    f = request.files['csv_file']
    if f.filename == '':
        return render_template("predict.html", csv_error="No file selected.")

    if not f.filename.lower().endswith('.csv'):
        return render_template("predict.html", csv_error="Please upload a valid .csv file.")

    try:
        csv_bytes = f.read()
        rows_data, csv_type, n_rows, col_info = extract_features_from_csv(csv_bytes)
    except ValueError as e:
        return render_template("predict.html", csv_error=str(e))
    except Exception as e:
        return render_template("predict.html", csv_error=f"Unexpected error: {e}")

    # Overall majority-vote summary
    all_svm = [r['svm'] for r in rows_data if r['svm'] in ('ADHD', 'CONTROL')]
    all_gcn = [r['gcn'] for r in rows_data if r['gcn'] in ('ADHD', 'CONTROL')]
    svm_summary = max(set(all_svm), key=all_svm.count) if all_svm else "N/A"
    gcn_summary = max(set(all_gcn), key=all_gcn.count) if all_gcn else "N/A"
    svm_adhd_pct = round(all_svm.count('ADHD') / len(all_svm) * 100, 1) if all_svm else 0
    gcn_adhd_pct = round(all_gcn.count('ADHD') / len(all_gcn) * 100, 1) if all_gcn else 0

    return render_template(
        "predict.html",
        csv_filename=f.filename,
        csv_type=csv_type,
        csv_col_info=col_info,
        csv_rows=rows_data,
        csv_n_rows=n_rows,
        csv_svm_summary=svm_summary,
        csv_gcn_summary=gcn_summary,
        csv_svm_adhd_pct=svm_adhd_pct,
        csv_gcn_adhd_pct=gcn_adhd_pct,
    )


def compute_df_band():
    records = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]
        group_type = "ADHD" if "ADHD" in mf.upper() else "CONTROL"

        for task_idx in range(cell.size):
            task = cell[0, task_idx]

            if isinstance(task, np.ndarray):
                if task.ndim == 2:
                    task = task[np.newaxis, :, :]

                n_subjects, n_samples, n_channels = task.shape
                ch_names = CHANNEL_NAMES[:n_channels]

                for s in range(n_subjects):
                    sig = task[s].astype(float)
                    if np.isnan(sig).any():
                        continue

                    for ch in range(n_channels):
                        x = sig[:, ch]

                        f, Pxx = sps.welch(x, fs=FS, nperseg=min(2048, max(256, len(x)//2)))
                        tot = np_trapz(Pxx, f)

                        row = {
                            "matfile": mf,
                            "task_idx": task_idx,
                            "subject_idx": s,
                            "channel": ch_names[ch],
                            "group_type": group_type
                        }

                        for b, (lo, hi) in BANDS.items():
                            idx = (f >= lo) & (f <= hi)
                            bp = np_trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0
                            row[f"{b}_abs"] = bp
                            row[f"{b}_rel"] = bp / tot if tot > 0 else 0.0

                        records.append(row)

    return pd.DataFrame(records)


print("Computing df_band...")
df_band = compute_df_band()
df_band = df_band.replace([np.inf, -np.inf], np.nan).fillna(0.0)
print("df_band ready:", df_band.shape)

# Coherence computation removed.
df_coh = pd.DataFrame()

# ADHD Evaluation disabled/removed as per user request.


@app.route("/subjects-per-group")
def subjects_per_group():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    records = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]
        group = group_names.get(mf, mf)

        for task_i in range(cell.size):
            task = cell[0, task_i]

            if isinstance(task, np.ndarray):
                n_sub = task.shape[0]
                records.append({"group": group, "matfile": mf, "subjects": n_sub})
                break

    df_subj = pd.DataFrame(records)

    fname = "subjects_per_group.png"
    fpath = os.path.join(out_dir, fname)

    plt.figure(figsize=(6,4))
    sns.barplot(data=df_subj, x="group", y="subjects")
    plt.title("Subjects Per Group")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()

    table_html = df_subj.to_html(classes="table", index=False)

    return render_template(
        "subjects_per_group.html",
        table=table_html,
        plot=fname
    )


@app.route("/bandpower-heatmap")
def bandpower_heatmap():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    bands = ["delta", "theta", "alpha", "beta", "gamma"]

    records = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        group = group_names.get(mf, mf)

        for task_i in range(cell.size):
            task = cell[0, task_i]

            if isinstance(task, np.ndarray):
                n_subj, n_samples, n_channels = task.shape

                for s in range(n_subj):
                    sig = task[s][:, 0]

                    f, Pxx = sps.welch(sig, fs=FS, nperseg=512)

                    row = {
                        "group": group,
                        "subject": f"{mf}_S{s}"
                    }

                    for b in bands:
                        lo, hi = {
                            "delta": (1,4),
                            "theta": (4,8),
                            "alpha": (8,13),
                            "beta": (13,30),
                            "gamma": (30,45)
                        }[b]

                        idx = (f >= lo) & (f <= hi)
                        bp = float(np_trapz(Pxx[idx], f[idx]))
                        row[b] = bp

                    records.append(row)

                break

    df = pd.DataFrame(records)

    heatmap_files = []

    for group in df["group"].unique():
        df_g = df[df["group"] == group]

        avg = df_g[bands].mean().to_frame(name="bandpower")

        fname = f"bandpower_heatmap_{group}.png"
        fpath = os.path.join(out_dir, fname)

        plt.figure(figsize=(4, 3))
        sns.heatmap(avg, annot=True, cmap="viridis")
        plt.title(f"Bandpower Heatmap â€” {group}")
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()

        heatmap_files.append(fname)

    table_html = df.head(15).to_html(classes="table", index=False)

    return render_template(
        "bandpower_heatmap.html",
        table=table_html,
        images=heatmap_files
    )


@app.route("/box-per-band")
def box_per_band():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    band_defs = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    rows = []

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]
        group = group_names.get(mf, mf)

        for t in range(cell.size):
            task = cell[0, t]

            if isinstance(task, np.ndarray):
                n_subj = task.shape[0]

                for s in range(n_subj):
                    sig = task[s][:, 0]
                    f, Pxx = sps.welch(sig, fs=FS, nperseg=512)

                    row = {"subject": f"{mf}_S{s}", "group": group}

                    for b in bands:
                        lo, hi = band_defs[b]
                        idx = (f >= lo) & (f <= hi)
                        row[b] = float(np_trapz(Pxx[idx], f[idx]))

                    rows.append(row)
                break

    df = pd.DataFrame(rows)

    plot_files = []

    for b in bands:
        fname = f"boxplot_{b}.png"
        fpath = os.path.join(out_dir, fname)

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="group", y=b)
        plt.title(f"{b.capitalize()} Bandpower by Group")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()

        plot_files.append(fname)

    table_html = df.head(15).to_html(classes="table", index=False)

    return render_template(
        "box_per_band.html",
        table=table_html,
        images=plot_files
    )







@app.route("/scalogram-gallery")
def scalogram_gallery():
    out_dir = os.path.join("static", "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_files = []

    MAX_IMAGES = 12   # Gallery limit â†’ prevents overload
    count = 0

    for mf, d in raw.items():
        key = next(iter(d.keys()))
        cell = d[key]

        group = group_names.get(mf, mf)

        for task_i in range(cell.size):
            task = cell[0, task_i]

            if isinstance(task, np.ndarray):

                n_subj = task.shape[0]

                for s in range(n_subj):

                    if count >= MAX_IMAGES:
                        break

                    sig = task[s][:, 0]

                    widths = np.arange(1, 128)
                    cwt_mat, freqs = pywt.cwt(sig, widths, 'cmor')

                    fname = f"scalogram_{group}_task{task_i}_S{s}.png"
                    fname = fname.replace(" ", "_")
                    fpath = os.path.join(out_dir, fname)

                    plt.figure(figsize=(7,5))
                    plt.imshow(
                        np.abs(cwt_mat),
                        extent=[0, len(sig)/FS, freqs[-1], freqs[0]],
                        cmap="viridis",
                        aspect="auto",
                        origin="upper"
                    )
                    plt.xlabel("Time (s)")
                    plt.ylabel("Frequency")
                    plt.title(f"{group} â€“ Task {task_i} â€“ Subj {s}")
                    plt.tight_layout()
                    plt.savefig(fpath)
                    plt.close()

                    plot_files.append(fname)
                    count += 1

                break

        if count >= MAX_IMAGES:
            break

    return render_template(
        "scalogram_gallery.html",
        images=plot_files
    )

# ---------------------------
# USER MANAGEMENT (CRUD)
# ---------------------------

@app.route("/users")
def users_list():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    users = conn.execute("SELECT id, username FROM users ORDER BY id ASC").fetchall()
    conn.close()

    return render_template("users_list.html", users=users)

@app.route("/users/add", methods=["GET", "POST"])
def users_add():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        confirm = request.form["confirm_password"].strip()

        if password != confirm:
            return render_template("users_add.html", error="Passwords do not match.")

        conn = get_db()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                         (username, password))
            conn.commit()
        except Exception as e:
            return render_template("users_add.html", error=str(e))
        conn.close()

        return redirect(url_for("users_list"))

    return render_template("users_add.html")


@app.route("/users/edit/<int:id>", methods=["GET", "POST"])
def users_edit(id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (id,)).fetchone()

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        confirm = request.form["confirm_password"].strip()

        if password != confirm:
            conn.close()
            return render_template("users_edit.html", user=user, error="Passwords do not match.")

        conn.execute(
            "UPDATE users SET username=?, password=? WHERE id=?",
            (username, password, id)
        )
        conn.commit()
        conn.close()

        return redirect(url_for("users_list"))

    conn.close()
    return render_template("users_edit.html", user=user)


@app.route("/users/delete/<int:id>", methods=["POST"])
def users_delete(id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect(url_for("users_list"))



if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)



