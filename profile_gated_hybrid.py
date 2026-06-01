import time
import random
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


INPUT_DIM = 433
OUTPUT_DIM = 512
NUM_RUNS = 100
WARMUP_RUNS = 30
SEED = 10024


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_single_frame_latency(model, inputs_100, device, warmup_runs=WARMUP_RUNS):
    model.eval()

    with torch.no_grad():
        for i in range(warmup_runs):
            x = inputs_100[i % inputs_100.shape[0]: i % inputs_100.shape[0] + 1].to(device)
            _ = model(x)
    sync_if_cuda(device)

    times = []
    with torch.no_grad():
        for i in range(inputs_100.shape[0]):
            x = inputs_100[i:i + 1].to(device)
            sync_if_cuda(device)
            start = time.perf_counter()
            _ = model(x)
            sync_if_cuda(device)
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times, dtype=np.float64)
    return {
        "avg_ms": float(times.mean() * 1000.0),
        "std_ms": float(times.std() * 1000.0),
        "min_ms": float(times.min() * 1000.0),
        "max_ms": float(times.max() * 1000.0),
        "fps": float(1.0 / times.mean()) if times.mean() > 0 else 0.0,
    }


def profile_major_ops_macs(model, sample_input):
    macs = {"total": 0}
    hooks = []

    def conv1d_hook(module, inputs, output):
        x = inputs[0]
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_len = output.shape[2]
        kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
        macs["total"] += int(batch_size * out_channels * out_len * kernel_ops)

    def linear_hook(module, inputs, output):
        # output shape: (..., out_features)
        macs["total"] += int(output.numel() * module.in_features)

    def mha_hook(module, inputs, output):
        # batch_first=True: q/k/v shape = [B, L, D]
        q = inputs[0]
        if q.dim() != 3:
            return
        batch_size, seq_len, embed_dim = q.shape
        num_heads = module.num_heads
        head_dim = embed_dim // num_heads

        qkv_proj = 3 * batch_size * seq_len * embed_dim * embed_dim
        attn_scores = batch_size * num_heads * seq_len * seq_len * head_dim
        attn_weighted = batch_size * num_heads * seq_len * seq_len * head_dim
        out_proj = batch_size * seq_len * embed_dim * embed_dim
        macs["total"] += int(qkv_proj + attn_scores + attn_weighted + out_proj)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))

    model.eval()
    with torch.no_grad():
        _ = model(sample_input)

    for h in hooks:
        h.remove()

    return int(macs["total"])


def print_result(model_name, params, macs, latency):
    flops = 2 * macs
    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print(f"Input shape:  (1, {INPUT_DIM})")
    print(f"Output shape: (1, {OUTPUT_DIM})")
    print("-" * 70)
    print(f"Trainable parameters: {params:,}")
    print(f"MACs:                 {macs:,}  ({macs / 1e6:.4f} MMACs)")
    print(f"FLOPs:                {flops:,}  ({flops / 1e6:.4f} MFLOPs)")
    print("-" * 70)
    print(f"Single-frame latency over {NUM_RUNS} curves:")
    print(f"Average: {latency['avg_ms']:.6f} ms")
    print(f"Std:     {latency['std_ms']:.6f} ms")
    print(f"Min:     {latency['min_ms']:.6f} ms")
    print(f"Max:     {latency['max_ms']:.6f} ms")
    print(f"FPS:     {latency['fps']:.2f} samples/s")
    print("=" * 70)


TRANS_D_MODEL = 64
TRANS_HEADS = 4
TRANS_LAYERS = 1


class GatedHybrid(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
        super(GatedHybrid, self).__init__()

        self.cnn_features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, TRANS_D_MODEL, 3, padding=1), nn.BatchNorm1d(TRANS_D_MODEL), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.trans_start = nn.Linear(input_dim, TRANS_D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, TRANS_D_MODEL))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANS_D_MODEL,
            nhead=TRANS_HEADS,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANS_LAYERS)
        self.trans_pool = nn.AdaptiveAvgPool1d(1)

        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(TRANS_D_MODEL, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        feat_cnn = self.cnn_features(x.unsqueeze(1)).flatten(1)

        xt = self.trans_start(x).unsqueeze(1) + self.pos_emb
        xt = self.transformer(xt)
        feat_trans = self.trans_pool(xt.permute(0, 2, 1)).flatten(1)

        beta = torch.sigmoid(self.gate_logit)
        feat_fused = beta * feat_cnn + (1.0 - beta) * feat_trans

        return self.fc(feat_fused)


if __name__ == "__main__":
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    model = GatedHybrid(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)
    model.eval()

    sample_input = torch.randn(1, INPUT_DIM, device=device)
    inputs_100 = torch.randn(NUM_RUNS, INPUT_DIM)

    params = count_trainable_params(model)
    macs = profile_major_ops_macs(model, sample_input)
    latency = measure_single_frame_latency(model, inputs_100, device)

    print_result("Gated Hybrid", params, macs, latency)
