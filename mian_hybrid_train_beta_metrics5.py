import os
import glob
import random
import copy
import sys
import datetime

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import joblib
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f'saved_models_hybrid_only_{current_time}'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


log_path = os.path.join(SAVE_DIR, "training_log.txt")
sys.stdout = Logger(log_path)



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n>>> [系统] 全局随机种子已固定为: {seed} <<<")


set_seed(10024)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
EPOCHS = 210
LR = 0.001

TRANS_D_MODEL = 64
TRANS_HEADS = 4
TRANS_LAYERS = 1

DATA_DIR = r'ASD_chuli/output_mats_grouped'
WAVE_FILE = r'ASD_chuli/waveband.csv'

ENABLE_AUGMENTATION = False
AUG_CONFIG = {
    'use_gain': True,
    'gain_range': (0.95, 1.05),
    'use_offset': False,
    'offset_ratio': 0.05,
    'use_noise': False,
    'noise_ratio': 0.02
}


def get_600nm_index(csv_path):
    if not os.path.exists(csv_path):
        raise ValueError(f"找不到波段文件: {csv_path}")
    df = pd.read_csv(csv_path, header=0)
    waves = df.iloc[:, 0].values.astype(float)
    target_nm = 600.0
    idx = (np.abs(waves - target_nm)).argmin()
    print(f"\n>>> [索引计算] 目标 600nm, 匹配到最近波长: {waves[idx]:.2f} nm, 索引位置: {idx}")
    return idx


try:
    INPUT_START_IDX = get_600nm_index(WAVE_FILE)
except Exception as e:
    print(f"读取波段文件失败: {e}")
    exit()


def augment_data(X, Y, config):
    print(f"\n>>> 正在进行数据增强...")
    n_samples, _ = X.shape
    X_aug = X.copy()
    Y_aug = Y.copy()

    if config['use_gain']:
        gain_factor = np.random.uniform(config['gain_range'][0], config['gain_range'][1], (n_samples, 1))
        X_aug *= gain_factor
        Y_aug *= gain_factor

    if config['use_offset']:
        std_x, std_y = np.std(X), np.std(Y)
        offset_x = np.random.normal(0, std_x * config['offset_ratio'], X.shape)
        offset_y = np.random.normal(0, std_y * config['offset_ratio'], Y.shape)
        X_aug += offset_x
        Y_aug += offset_y

    if config['use_noise']:
        std_x, std_y = np.std(X), np.std(Y)
        noise_x = np.random.normal(0, std_x * config['noise_ratio'], X.shape)
        noise_y = np.random.normal(0, std_y * config['noise_ratio'], Y.shape)
        X_aug += noise_x
        Y_aug += noise_y

    return np.vstack((X, X_aug)), np.vstack((Y, Y_aug))


def load_data_from_folder(folder_path):
    print(f"正在扫描文件夹: {folder_path}")
    search_path = os.path.join(folder_path, '*_interp.mat')
    file_list = glob.glob(search_path)
    if not file_list:
        raise ValueError("未找到数据文件")

    all_data = []
    for fname in file_list:
        try:
            mat = sio.loadmat(fname)
            data = mat.get('train_data', mat.get('raw_data'))
            if data is None:
                valid_keys = [k for k in mat.keys() if not k.startswith('__')]
                if valid_keys:
                    data = mat[valid_keys[0]]

            if data is not None and data.ndim == 2:
                all_data.append(data.astype(np.float32))
        except Exception as e:
            print(f"跳过文件 {fname}: {e}")
            continue

    if not all_data:
        raise ValueError("加载失败")
    full_dataset = np.concatenate(all_data, axis=0)

    if INPUT_START_IDX >= full_dataset.shape[1]:
        raise ValueError("索引越界")

    X = full_dataset[:, INPUT_START_IDX:]
    Y = full_dataset[:, :]

    print(f"-> 原始数据加载成功，维度: {X.shape}")

    if ENABLE_AUGMENTATION:
        X, Y = augment_data(X, Y, AUG_CONFIG)
        print(f"-> 增强后数据维度: {X.shape}")
    else:
        print("-> 数据增强已关闭")

    return X, Y


if not os.path.exists(DATA_DIR):
    print("数据目录不存在")
    exit()

X_raw_all, Y_raw_all = load_data_from_folder(DATA_DIR)

X_train_val, X_test_raw, Y_train_val, Y_test_raw = train_test_split(
    X_raw_all, Y_raw_all, test_size=0.2, random_state=42, shuffle=True
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.25, random_state=42, shuffle=True
)

print(f"数据集划分 -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test_raw.shape[0]}")

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_x.fit_transform(X_train)
Y_train = scaler_y.fit_transform(Y_train)
X_val = scaler_x.transform(X_val)
Y_val = scaler_y.transform(Y_val)
X_test_scaled = scaler_x.transform(X_test_raw)

# DataLoader
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
    batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test_scaled)),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class GatedHybrid(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedHybrid, self).__init__()


        self.cnn_features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, TRANS_D_MODEL, 3, padding=1), nn.BatchNorm1d(TRANS_D_MODEL), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.trans_start = nn.Linear(input_dim, TRANS_D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, TRANS_D_MODEL))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANS_D_MODEL,
            nhead=TRANS_HEADS,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANS_LAYERS)

        self.trans_pool = nn.AdaptiveAvgPool1d(1)


        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(TRANS_D_MODEL, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def get_beta(self):
        return torch.sigmoid(self.gate_logit.detach()).item()

    def forward(self, x):
        feat_cnn = self.cnn_features(x.unsqueeze(1)).flatten(1)

        xt = self.trans_start(x).unsqueeze(1) + self.pos_emb
        xt = self.transformer(xt)
        feat_trans = self.trans_pool(xt.permute(0, 2, 1)).flatten(1)

        beta = torch.sigmoid(self.gate_logit)
        feat_fused = beta * feat_cnn + (1 - beta) * feat_trans

        return self.fc(feat_fused)

def calc_rmse_batch(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calc_mae_batch(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean(np.abs(y_true - y_pred))

def calc_sam_batch(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1) + 1e-9
    cos_theta = np.clip(np.sum(y_true * y_pred, axis=1) / denom, -1.0, 1.0)
    return np.mean(np.arccos(cos_theta))

def calc_sid_batch(y_true, y_pred):
    eps = 1e-12
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    p = np.clip(y_true, eps, None)
    q = np.clip(y_pred, eps, None)
    p = p / (np.sum(p, axis=1, keepdims=True) + eps)
    q = q / (np.sum(q, axis=1, keepdims=True) + eps)

    sid_pq = np.sum(p * np.log((p + eps) / (q + eps)), axis=1)
    sid_qp = np.sum(q * np.log((q + eps) / (p + eps)), axis=1)
    return np.mean(sid_pq + sid_qp)


def get_metrics_batch(y_true, y_pred):
    rmse = calc_rmse_batch(y_true, y_pred)
    sam = calc_sam_batch(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = calc_mae_batch(y_true, y_pred)
    sid = calc_sid_batch(y_true, y_pred)
    return rmse, sam, r2, mae, sid


def train_model(model, name):
    print("\n" + "=" * 60 + f"\n   MODEL STRUCTURE: {name}\n" + "=" * 60)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 60 + f"\n   >>> Total Trainable Parameters: {total_params:,}\n" + "=" * 60 + "\n")

    print(f"=== 正在训练 {name} ===")
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_v_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_beta = model.get_beta()

    for ep in range(EPOCHS):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            v_loss = np.mean([criterion(model(x.to(DEVICE)), y.to(DEVICE)).item() for x, y in val_loader])

        current_beta = model.get_beta()
        gate_info = f" | beta(CNN): {current_beta:.8f} | beta(Transformer): {1-current_beta:.8f}"

        if v_loss < best_v_loss:
            best_v_loss = v_loss
            best_epoch = ep + 1
            best_beta = current_beta
            best_model_wts = copy.deepcopy(model.state_dict())
            save_msg = " [Saving Best Model]"
        else:
            save_msg = ""

        print(
            f"Epoch {ep + 1}/{EPOCHS} | Train: {np.mean(batch_losses):.4f} | Val: {v_loss:.4f}{gate_info}{save_msg}"
        )

    print(f"\n>>> {name} 训练结束。最佳 Epoch: {best_epoch}, 最佳 Val Loss: {best_v_loss:.4f}")
    model.load_state_dict(best_model_wts)
    final_beta = model.get_beta()
    print(f">>> [最佳模型] beta(CNN 权重) = {final_beta:.8f}")
    print(f">>> [最佳模型] 1 - beta(Transformer 权重) = {1 - final_beta:.8f}")

    beta_path = os.path.join(SAVE_DIR, 'final_beta.txt')
    with open(beta_path, 'w', encoding='utf-8') as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={best_v_loss:.10f}\n")
        f.write(f"beta_cnn={final_beta:.10f}\n")
        f.write(f"beta_transformer={1-final_beta:.10f}\n")
    print(f">>> beta 已保存到: {beta_path}")

    return model


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            out = model(x)
            preds.append(out.cpu().numpy())
    return scaler_y.inverse_transform(np.concatenate(preds, axis=0))


def evaluate_dataset(dataset_name, y_true, loader_for_dl):
    print(f"\n>>>>>> 正在评估数据集: {dataset_name} <<<<<<")

    y_hybrid = predict(hybrid_model, loader_for_dl)
    rmse, sam, r2, mae, sid = get_metrics_batch(y_true, y_hybrid)

    print(
        f"[Gated Hybrid] RMSE: {rmse:.6f} | SAM: {sam:.6f} | R2: {r2:.6f} | MAE: {mae:.6f} | SID: {sid:.6f}"
    )

    metrics_df = pd.DataFrame([{
        'Dataset': dataset_name,
        'Model': 'Gated Hybrid',
        'RMSE': rmse,
        'SAM': sam,
        'R2': r2,
        'MAE': mae,
        'SID': sid,
        'Beta_CNN': hybrid_model.get_beta(),
        'Beta_Transformer': 1 - hybrid_model.get_beta()
    }])
    safe_name = dataset_name.replace(' ', '_').replace('/', '_')
    metrics_path = os.path.join(SAVE_DIR, f'{safe_name}_hybrid_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"-> 指标已保存为: {metrics_path}")

    return {'Gated Hybrid': y_hybrid}, metrics_df


hybrid_model = train_model(GatedHybrid(X_train.shape[1], Y_train.shape[1]), "Gated Hybrid")

print("\n" + "=" * 50 + "\n   TEST SET: 原始数据 (Raw Data, No Smoothing)\n" + "=" * 50)
results_raw, metrics_raw = evaluate_dataset("Raw Data", Y_test_raw, test_loader)

metrics_all = metrics_raw.copy()
metrics_all_path = os.path.join(SAVE_DIR, 'hybrid_metrics_all.csv')
metrics_all.to_csv(metrics_all_path, index=False)
print(f"\n>>> 所有 Hybrid 指标汇总已保存为: {metrics_all_path}")


def plot_and_export_6_samples(y_true, preds_dict, start_idx=INPUT_START_IDX):
    print("\n>>> 正在生成 6 个随机样本的 Hybrid 对比图及导出原始数据...")

    num_samples = y_true.shape[0]
    n_plot = min(6, num_samples)
    random_indices = np.random.choice(num_samples, n_plot, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    all_csv_data = []

    for i, idx in enumerate(random_indices):
        ax = axes[i]
        gt = y_true[idx]
        hybrid_pred = preds_dict["Gated Hybrid"][idx]

        ax.plot(gt, 'k', linewidth=2, label='Ground Truth')
        ax.plot(hybrid_pred, 'r-', linewidth=2, alpha=0.8, label='Gated Hybrid')
        ax.axvline(start_idx, color='r', linestyle=':', alpha=0.5, label='Input Start')

        ax.set_title(f'Test Sample ID: {idx}')
        ax.set_xlabel('Bands')
        ax.set_ylabel('Reflectance')
        if i == 0:
            ax.legend(loc='best')

        points = np.arange(len(gt))
        df_sample = pd.DataFrame({
            'Sample_ID': idx,
            'Band_Index': points,
            'Ground_Truth': gt,
            'Gated_Hybrid': hybrid_pred
        })
        all_csv_data.append(df_sample)

    for j in range(len(random_indices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    img_path = os.path.join(SAVE_DIR, '6_random_samples_hybrid_comparison.png')
    plt.savefig(img_path, dpi=300)
    plt.close(fig)
    print(f"-> 图像已保存为: {img_path}")

    final_df = pd.concat(all_csv_data, ignore_index=True)
    csv_path = os.path.join(SAVE_DIR, '6_random_samples_hybrid_data.csv')
    final_df.to_csv(csv_path, index=False)
    print(f"-> 6个样本的数据已保存为: {csv_path}")


def save_all_test_predictions(y_true, preds_dict):
    save_data = {'Ground_Truth': y_true}
    save_data.update(preds_dict)
    npz_path = os.path.join(SAVE_DIR, 'all_test_results_hybrid_only.npz')
    np.savez(npz_path, **save_data)
    print(f"\n>>> 所有测试集 Hybrid 预测结果已打包保存为: {npz_path}")


plot_and_export_6_samples(Y_test_raw, results_raw)
save_all_test_predictions(Y_test_raw, results_raw)

def save_models():
    print(f"\n>>> 正在保存 Hybrid 模型权重与归一化器到 '{SAVE_DIR}' ...")

    torch.save(hybrid_model.state_dict(), os.path.join(SAVE_DIR, 'hybrid.pth'))
    joblib.dump(scaler_x, os.path.join(SAVE_DIR, 'scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(SAVE_DIR, 'scaler_y.pkl'))
    print(">>> Hybrid 模型与 Scaler 保存完成！")


save_models()
