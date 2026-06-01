import os
import torch
import joblib
import scipy.io as sio
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096

SAVE_DIR = './compare_results_hybrid_xxx'
MODEL_DIR = 'saved_models_final_xxx'

PATH_HYBRID = os.path.join(MODEL_DIR, 'hybrid.pth')
PATH_SCALER_X = os.path.join(MODEL_DIR, 'scaler_x.pkl')
PATH_SCALER_Y = os.path.join(MODEL_DIR, 'scaler_y.pkl')
PATH_BETA_TXT = os.path.join(MODEL_DIR, 'final_beta.txt')

MAT_FILE_PATH = r'MAT_data/20230308_c_20hz_3300us_XXX.mat'
WAVE_FILE = r'ASD_chuli/waveband.csv'


TRANS_D_MODEL = 64
TRANS_HEADS = 4
TRANS_LAYERS = 1

LEGACY_DEFAULT_FIXED_BETA = 0.5


class GatedHybrid(nn.Module):
    def __init__(self, input_dim, output_dim, fixed_beta=LEGACY_DEFAULT_FIXED_BETA):
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
        self.register_buffer('fixed_beta', torch.tensor(float(fixed_beta)))
        self.use_fixed_beta = False

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(TRANS_D_MODEL, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def set_beta_mode_from_state_dict(self, state_dict):
        if 'fixed_beta' in state_dict:
            self.use_fixed_beta = True
            return 'fixed_beta'
        if 'gate_logit' in state_dict:
            self.use_fixed_beta = False
            return 'trainable_beta'
        self.use_fixed_beta = True
        return 'legacy_fixed_beta_default'

    def get_beta(self):
        if self.use_fixed_beta:
            return float(self.fixed_beta.detach().cpu().item())
        return float(torch.sigmoid(self.gate_logit.detach()).cpu().item())

    def forward(self, x):
        feat_cnn = self.cnn_features(x.unsqueeze(1)).flatten(1)

        xt = self.trans_start(x).unsqueeze(1) + self.pos_emb
        xt = self.transformer(xt)
        feat_trans = self.trans_pool(xt.permute(0, 2, 1)).flatten(1)

        if self.use_fixed_beta:
            beta = self.fixed_beta.to(x.device)
        else:
            beta = torch.sigmoid(self.gate_logit)
        feat_fused = beta * feat_cnn + (1 - beta) * feat_trans

        return self.fc(feat_fused)


def check_required_files():
    required_paths = {
        'Hybrid 权重': PATH_HYBRID,
        'Scaler X': PATH_SCALER_X,
        'Scaler Y': PATH_SCALER_Y,
        '波段文件': WAVE_FILE,
        '待测试 MAT 数据': MAT_FILE_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(
            "以下必要文件不存在，请先修改脚本配置区的 MODEL_DIR / MAT_FILE_PATH / WAVE_FILE：\n" + msg
        )


def get_600nm_index(csv_path):
    df = pd.read_csv(csv_path, header=0)
    waves = df.iloc[:, 0].values.astype(float)
    idx = int((np.abs(waves - 600.0)).argmin())
    print(f">>> 目标 600nm，匹配到最近波长: {waves[idx]:.2f} nm，索引位置: {idx}")
    return idx, waves


def load_image(mat_path):
    print(f">>> Loading MAT file: {mat_path}")
    try:
        mat = sio.loadmat(mat_path)
        valid_keys = [k for k in mat.keys() if not k.startswith('__')]
        if not valid_keys:
            raise ValueError("MAT 文件中没有找到有效变量")
        data = mat[max(valid_keys, key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0)]
    except NotImplementedError:
        print("    [Info] Detected v7.3 format, using h5py...")
        with h5py.File(mat_path, 'r') as f:
            dataset_keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            if not dataset_keys:
                raise ValueError("HDF5 MAT 文件中没有找到有效 Dataset")
            best_key = max(dataset_keys, key=lambda k: f[k].size)
            data = np.transpose(f[best_key][()], (2, 1, 0))

    if data.ndim != 3:
        raise ValueError(f"读取到的数据不是三维高光谱数据，当前维度: {data.shape}")
    return data


def generate_rgb(cube, waves, r_nm=640, g_nm=550, b_nm=460):
    idx = [int(np.abs(waves - nm).argmin()) for nm in [r_nm, g_nm, b_nm]]
    rgb = np.stack([cube[:, :, i] for i in idx], axis=-1)
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if p98 - p2 > 0:
        rgb = (rgb - p2) / (p98 - p2)
    return np.clip(rgb, 0, 1)


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


def calc_metrics_5(y_true, y_pred):
    rmse = calc_rmse_batch(y_true, y_pred)
    sam = calc_sam_batch(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = calc_mae_batch(y_true, y_pred)
    sid = calc_sid_batch(y_true, y_pred)
    return rmse, sam, r2, mae, sid


def predict_dl(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_state_dict_compat(path):
    try:
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=DEVICE)

    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v
    return cleaned


def read_beta_txt_if_exists(path):
    if not os.path.exists(path):
        return None
    info = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                info[k.strip()] = v.strip()
    return info


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    check_required_files()

    input_start_idx, waves = get_600nm_index(WAVE_FILE)
    raw_cube = load_image(MAT_FILE_PATH).astype(np.float32)
    H, W, C = raw_cube.shape
    print(f">>> Raw cube shape: H={H}, W={W}, Bands={C}")
    input_cube = raw_cube

    flat_data = input_cube.reshape(-1, C)
    scaler_x = joblib.load(PATH_SCALER_X)
    scaler_y = joblib.load(PATH_SCALER_Y)
    X_input_scaled = scaler_x.transform(flat_data[:, input_start_idx:])

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_input_scaled)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    input_dim = X_input_scaled.shape[1]

    print(f">>> Loading Hybrid model: {PATH_HYBRID}")
    state_dict = load_state_dict_compat(PATH_HYBRID)
    model = GatedHybrid(input_dim, C).to(DEVICE)
    beta_mode = model.set_beta_mode_from_state_dict(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    ignorable_missing = {'gate_logit', 'fixed_beta'}
    important_missing = [k for k in missing_keys if k not in ignorable_missing]
    if important_missing:
        print(f"[Warning] 加载权重时存在未匹配的重要参数: {important_missing}")
    if unexpected_keys:
        print(f"[Warning] 加载权重时存在额外参数: {unexpected_keys}")

    beta = model.get_beta()
    print(f">>> β 模式: {beta_mode}")
    print(f">>> 当前 beta(CNN 权重) = {beta:.10f}")
    print(f">>> 当前 1 - beta(Transformer 权重) = {1 - beta:.10f}")

    beta_txt = read_beta_txt_if_exists(PATH_BETA_TXT)
    if beta_txt:
        print(f">>> final_beta.txt 信息: {beta_txt}")

    print(">>> Running Hybrid inference...")
    pred_scaled = predict_dl(model, loader)
    hybrid_cube = scaler_y.inverse_transform(pred_scaled).reshape(H, W, C).astype(np.float32)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70 + "\n   Hybrid Evaluation Metrics\n" + "=" * 70)
    flat_gt = raw_cube.reshape(-1, C)
    flat_pred = hybrid_cube.reshape(-1, C)
    rmse, sam, r2, mae, sid = calc_metrics_5(flat_gt, flat_pred)
    print(
        f"[Hybrid] RMSE: {rmse:.8f} | SAM: {sam:.8f} | R2: {r2:.8f} | "
        f"MAE: {mae:.8f} | SID: {sid:.8f}"
    )

    metrics_df = pd.DataFrame([{
        'Model': 'Hybrid',
        'Beta_mode': beta_mode,
        'Beta_CNN': beta,
        'Beta_Transformer': 1 - beta,
        'RMSE': rmse,
        'SAM': sam,
        'R2': r2,
        'MAE': mae,
        'SID': sid,
    }])
    metrics_path = os.path.join(SAVE_DIR, 'hybrid_metrics_5.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f">>> 5个指标已保存: {metrics_path}")

    plot_list = [('Ground Truth', raw_cube), ('Hybrid', hybrid_cube)]
    n_cols = len(plot_list)

    print("\n>>> Plotting RGB comparison...")
    plt.figure(figsize=(4 * n_cols, 5))
    for i, (name, cube) in enumerate(plot_list):
        plt.subplot(1, n_cols, i + 1)
        plt.imshow(generate_rgb(cube, waves))
        plt.title(name, fontsize=14, pad=10)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.suptitle("RGB Composite Comparison (R:640nm, G:550nm, B:460nm)", fontsize=16, y=0.98)
    rgb_path = os.path.join(SAVE_DIR, 'compare_rgb_hybrid_only.png')
    plt.savefig(rgb_path, dpi=300)
    plt.close()
    print(f"    Saved: {rgb_path}")

    print(">>> Plotting 450nm comparison...")
    target_idx = int(np.abs(waves - 450.0).argmin())
    plt.figure(figsize=(4 * n_cols, 5))
    for i, (name, cube) in enumerate(plot_list):
        plt.subplot(1, n_cols, i + 1)
        plt.imshow(cube[:, :, target_idx], cmap='gray')
        plt.title(name, fontsize=14, pad=10)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.suptitle("Single Band Reconstruction @ 450nm (Blind Region)", fontsize=16, y=0.98)
    band_path = os.path.join(SAVE_DIR, 'compare_450nm_hybrid_only.png')
    plt.savefig(band_path, dpi=300)
    plt.close()
    print(f"    Saved: {band_path}")

    print(">>> Plotting spectral curves...")
    rng = np.random.RandomState(42)
    sample_points = [(rng.randint(0, H), rng.randint(0, W)) for _ in range(4)]
    styles = {
        'Ground Truth': ('k-', 2, 0.6),
        'Hybrid': ('r-', 1.5, 0.9),
    }

    plt.figure(figsize=(16, 12))
    for i, (r, c) in enumerate(sample_points):
        plt.subplot(2, 2, i + 1)
        for name, cube in plot_list:
            s = styles[name]
            plt.plot(waves, cube[r, c, :], s[0], linewidth=s[1], alpha=s[2], label=name)
        plt.axvline(waves[input_start_idx], color='gray', linestyle=':', label='Input Start')
        plt.title(f'Pixel ({r}, {c})', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Spectral Curve Comparison: Ground Truth vs Hybrid", fontsize=16)
    spectra_path = os.path.join(SAVE_DIR, 'compare_spectra_hybrid_only.png')
    plt.savefig(spectra_path, dpi=300)
    plt.close()
    print(f"    Saved: {spectra_path}")

    print("\n>>> Saving results to individual .mat files...")
    gt_path = os.path.join(SAVE_DIR, 'result_GroundTruth.mat')
    hybrid_path = os.path.join(SAVE_DIR, 'result_Hybrid.mat')
    sio.savemat(gt_path, {'GroundTruth': raw_cube}, do_compression=True)
    sio.savemat(hybrid_path, {'Hybrid': hybrid_cube}, do_compression=True)
    print(f"    Saved: {gt_path}")
    print(f"    Saved: {hybrid_path}")

    beta_save_path = os.path.join(SAVE_DIR, 'test_beta_info.txt')
    with open(beta_save_path, 'w', encoding='utf-8') as f:
        f.write(f"beta_mode={beta_mode}\n")
        f.write(f"beta_cnn={beta:.10f}\n")
        f.write(f"beta_transformer={1-beta:.10f}\n")
        f.write(f"model_dir={MODEL_DIR}\n")
        f.write(f"path_hybrid={PATH_HYBRID}\n")
    print(f"    Saved: {beta_save_path}")

    print("\n>>> All tasks completed.")


if __name__ == '__main__':
    main()
