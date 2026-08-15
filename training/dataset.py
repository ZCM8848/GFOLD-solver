"""数据集加载与切分。

- 读 data/*.npz 全部 shard，拼成 X / y_feasible / y_tf
- X 按 meta.json 的 NORM_BOUNDS 做 min-max 归一化到 [0,1]
- TF 用可行训练样本的 mean/std 做 z-score（不可行样本填 0，训练时用 mask 屏蔽）
- 按 feasible 标签分层切 train/val/test
- 两种消费方式：CPU DataLoader，或 GPU 显存预加载批迭代器（零逐批 H2D 搬运）
"""

import glob
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from common import config as cfg


class TOFNetDataset(Dataset):
    def __init__(self, X, y_feas, y_tf):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y_feas = torch.as_tensor(y_feas, dtype=torch.float32)
        self.y_tf = torch.as_tensor(y_tf, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y_feas[i], self.y_tf[i]


class GPUBatches:
    """整个 split 预加载进显存，按批切片 yield GPU 张量（无逐批 H2D 搬运）。"""

    def __init__(self, X, y_feas, y_tf, batch_size, device, shuffle):
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.y_feas = torch.as_tensor(y_feas, dtype=torch.float32, device=device)
        self.y_tf = torch.as_tensor(y_tf, dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.n = self.X.shape[0]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n, bs = self.n, self.batch_size
        idx = (torch.randperm(n, device=self.device) if self.shuffle
               else torch.arange(n, device=self.device))
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            yield self.X[b], self.y_feas[b], self.y_tf[b]


def load_raw(datadir):
    meta_path = os.path.join(datadir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]
    lo = np.array([meta["norm_bounds"][name][0] for name in features], dtype=np.float32)
    hi = np.array([meta["norm_bounds"][name][1] for name in features], dtype=np.float32)

    shards = sorted(glob.glob(os.path.join(datadir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"{datadir} 下没有 shard_*.npz")

    Xs, Yf, Yt = [], [], []
    for s in shards:
        d = np.load(s)
        Xs.append(d["X"])
        Yf.append(d["y_feasible"])
        Yt.append(d["y_tf"])
    X = np.concatenate(Xs).astype(np.float32)
    y_feas = np.concatenate(Yf).astype(np.float32)
    y_tf = np.concatenate(Yt).astype(np.float32)
    return X, y_feas, y_tf, lo, hi, meta


def prepare_splits(datadir, val_frac=0.1, test_frac=0.1, seed=0, max_samples=None):
    X, y_feas, y_tf, lo, hi, meta = load_raw(datadir)
    if max_samples is not None:
        X = X[:max_samples]
        y_feas = y_feas[:max_samples]
        y_tf = y_tf[:max_samples]

    Xn = (X - lo) / (hi - lo)          # min-max -> [0,1]

    idx = np.arange(len(Xn))
    trval_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=y_feas, random_state=seed)
    val_frac_of_rest = val_frac / (1.0 - test_frac)
    tr_idx, val_idx = train_test_split(
        trval_idx, test_size=val_frac_of_rest,
        stratify=y_feas[trval_idx], random_state=seed)

    # TF z-score（只用训练集可行样本估计 mean/std）
    tf_feas = y_tf[tr_idx][y_feas[tr_idx].astype(bool)]
    tf_mean = float(tf_feas.mean())
    tf_std = float(tf_feas.std())
    y_tf_std = np.where(np.isnan(y_tf), 0.0, (y_tf - tf_mean) / tf_std).astype(np.float32)

    splits = {}
    for name, ids in [("train", tr_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = (Xn[ids], y_feas[ids], y_tf_std[ids])

    stats = {
        "tf_mean": tf_mean, "tf_std": tf_std,
        "lo": lo, "hi": hi, "meta": meta,
        "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    return splits, stats


def build_loaders(datadir, val_frac=0.1, test_frac=0.1, seed=0,
                  batch_size=1024, num_workers=0, max_samples=None):
    splits, stats = prepare_splits(datadir, val_frac, test_frac, seed, max_samples)
    loaders = {}
    for name, (X, yf, yt) in splits.items():
        ds = TOFNetDataset(X, yf, yt)
        loaders[name] = DataLoader(ds, batch_size=batch_size,
                                   shuffle=(name == "train"), num_workers=num_workers)
    return loaders, stats


def build_gpu_batches(datadir, device, val_frac=0.1, test_frac=0.1, seed=0,
                      batch_size=1024, max_samples=None):
    splits, stats = prepare_splits(datadir, val_frac, test_frac, seed, max_samples)
    batches = {}
    for name, (X, yf, yt) in splits.items():
        batches[name] = GPUBatches(X, yf, yt, batch_size, device,
                                   shuffle=(name == "train"))
    return batches, stats
