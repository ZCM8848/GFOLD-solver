"""TOF-Net 训练脚本。

损失：L = BCE(可行性, 全样本) + λ·MSE(TF, 仅可行样本, z-score)
指标：PR-AUC / accuracy / TF MAE·RMSE（可行样本，反标准化回秒）
产出：models/tofnet.pt（权重+契约）+ models/tofnet.json（人类可读）

用法：
    python -m training.train                      # 全量训练
    python -m training.train --max-samples 50000 --epochs 2   # 冒烟测试
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from common import config as cfg
from common.model import TOFNet
from training import dataset as ds


def combined_loss(logits, tf_pred, y_feas, y_tf, lam=1.0):
    bce = F.binary_cross_entropy_with_logits(logits.view(-1), y_feas)
    sq = ((tf_pred.view(-1) - y_tf) ** 2) * y_feas
    mse = sq.sum() / (y_feas.sum() + 1e-8)
    return bce + lam * mse, bce.item(), mse.item()


def evaluate(model, loader, stats, device, lam=1.0):
    model.eval()
    probs, feas_true, tf_true, tf_pred = [], [], [], []
    tot_bce, tot_mse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, yf, yt in loader:
            x, yf, yt = x.to(device), yf.to(device), yt.to(device)
            logits, tfp = model(x)
            bce = F.binary_cross_entropy_with_logits(logits.view(-1), yf)
            mse = ((tfp.view(-1) - yt) ** 2 * yf).sum() / (yf.sum() + 1e-8)
            tot_bce += float(bce)
            tot_mse += float(mse)
            n += 1
            probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            feas_true.append(yf.cpu().numpy().ravel())
            tf_true.append(yt.cpu().numpy().ravel())
            tf_pred.append(tfp.cpu().numpy().ravel())

    probs = np.concatenate(probs)
    feas_true = np.concatenate(feas_true).astype(bool)
    tf_true = np.concatenate(tf_true)
    tf_pred = np.concatenate(tf_pred)

    ap = float(average_precision_score(feas_true, probs))
    acc = float(np.mean((probs >= 0.5) == feas_true))

    mask = feas_true
    tt = tf_true[mask] * stats["tf_std"] + stats["tf_mean"]
    tp = tf_pred[mask] * stats["tf_std"] + stats["tf_mean"]
    mae = float(np.mean(np.abs(tt - tp)))
    rmse = float(np.sqrt(np.mean((tt - tp) ** 2)))

    return {
        "loss": (tot_bce + lam * tot_mse) / n,
        "bce": tot_bce / n, "mse": tot_mse / n,
        "ap": ap, "acc": acc, "mae": mae, "rmse": rmse,
        "probs": probs, "feas": feas_true,
    }


def select_threshold(probs, feas, beta=1.0):
    """在 PR 曲线上按 F_beta 选阈值（beta<1 偏向 precision，即少误报可行）。"""
    order = np.argsort(probs)[::-1]
    p = probs[order]
    f = feas[order]
    tp = np.cumsum(f)
    fp = np.cumsum(~f)
    fn = f.sum() - tp
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / np.maximum(tp + fn, 1e-9)
    fbeta = ((1 + beta ** 2) * precision * recall
             / np.maximum(beta ** 2 * precision + recall, 1e-9))
    best = int(np.argmax(fbeta))
    return float(p[best]), float(fbeta[best])


def save_checkpoint(model, stats, args, threshold, metrics, path):
    ckpt = {
        "state_dict": model.state_dict(),
        "in_features": cfg.N_FEATURES,
        "hidden": list(args.hidden),
        "activation": args.act,
        "dropout": args.dropout,
        "features": cfg.FEATURES,
        "norm_bounds": {k: list(v) for k, v in cfg.NORM_BOUNDS.items()},
        "tf_mean": stats["tf_mean"],
        "tf_std": stats["tf_std"],
        "threshold": threshold,
        "hyperparams": {k: getattr(args, k) for k in
                        ["lr", "wd", "batch", "lam", "epochs", "seed"]},
    }
    torch.save(ckpt, path)
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    meta["metrics"] = metrics
    json_path = path.replace(".pt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return json_path


def parse_args():
    p = argparse.ArgumentParser(description="TOF-Net 训练")
    p.add_argument("--datadir", default="data")
    p.add_argument("--outdir", default="models")
    p.add_argument("--logdir", default="logs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=1.0, help="回归损失权重 λ")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 64])
    p.add_argument("--act", default="silu", choices=["silu", "relu", "gelu"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None, help="限制样本数（冒烟测试用）")
    p.add_argument("--beta", type=float, default=1.0, help="阈值选择的 F_beta")
    p.add_argument("--threads", type=int, default=0, help="torch 线程数，0=自动")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="训练设备，auto=有 GPU 就用 GPU")
    p.add_argument("--no-preload-gpu", action="store_true",
                   help="GPU 训练时不把数据预加载进显存（默认预加载）")
    return p.parse_args()


def resolve_device(pref):
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda 但未检测到 CUDA")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args()
    threads = args.threads or (os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)} "
              f"(torch {torch.__version__}, cuda {torch.version.cuda})")

    print(f"[load] 读数据集 {args.datadir} ...")
    preload = (device.type == "cuda") and not args.no_preload_gpu
    if preload:
        try:
            loaders, stats = ds.build_gpu_batches(
                args.datadir, device, val_frac=args.val, test_frac=args.test,
                seed=args.seed, batch_size=args.batch, max_samples=args.max_samples)
            print("[data] 数据已预加载进显存（每 epoch 无逐批搬运）")
        except RuntimeError as e:
            print(f"[data] GPU 预加载失败（{e}），回退 CPU DataLoader")
            loaders, stats = ds.build_loaders(
                args.datadir, val_frac=args.val, test_frac=args.test, seed=args.seed,
                batch_size=args.batch, num_workers=args.num_workers,
                max_samples=args.max_samples)
    else:
        loaders, stats = ds.build_loaders(
            args.datadir, val_frac=args.val, test_frac=args.test, seed=args.seed,
            batch_size=args.batch, num_workers=args.num_workers,
            max_samples=args.max_samples)
    print(f"[data] train={stats['n_train']:,} val={stats['n_val']:,} "
          f"test={stats['n_test']:,}  tf_mean={stats['tf_mean']:.2f}s "
          f"tf_std={stats['tf_std']:.2f}s")

    model = TOFNet(in_features=cfg.N_FEATURES, hidden=args.hidden,
                   dropout=args.dropout, activation=args.act).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数 {n_params:,}  hidden={args.hidden}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "tofnet.pt")

    os.makedirs(args.logdir, exist_ok=True)
    csv_file = open(os.path.join(args.logdir, "train_metrics.csv"), "w",
                    newline="", encoding="utf-8")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["epoch", "lr", "train_loss", "train_bce", "train_mse",
                    "val_loss", "val_ap", "val_acc", "val_mae", "val_rmse"])

    best_loss = float("inf")
    best_state = None
    best_val_probs = None
    best_val_feas = None
    best_metrics = None

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss, run_bce, run_mse, n = 0.0, 0.0, 0.0, 0
        for x, yf, yt in loaders["train"]:
            x, yf, yt = x.to(device), yf.to(device), yt.to(device)
            logits, tfp = model(x)
            loss, bce, mse = combined_loss(logits, tfp, yf, yt, lam=args.lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item()
            run_bce += bce
            run_mse += mse
            n += 1
        sched.step()

        vm = evaluate(model, loaders["val"], stats, device, lam=args.lam)
        csv_w.writerow([epoch, opt.param_groups[0]["lr"],
                        run_loss / n, run_bce / n, run_mse / n,
                        vm["loss"], vm["ap"], vm["acc"], vm["mae"], vm["rmse"]])
        csv_file.flush()
        line = (f"epoch {epoch:3d}/{args.epochs}  "
                f"train {run_loss/n:.4f} (bce {run_bce/n:.4f} mse {run_mse/n:.4f})  "
                f"val loss {vm['loss']:.4f}  AP {vm['ap']:.4f}  acc {vm['acc']:.4f}  "
                f"MAE {vm['mae']:.3f}s  RMSE {vm['rmse']:.3f}s  "
                f"[{time.time()-t0:.1f}s]")
        print(line, flush=True)

        if vm["loss"] < best_loss:
            best_loss = vm["loss"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val_probs = vm["probs"]
            best_val_feas = vm["feas"]
            best_metrics = {k: vm[k] for k in ["ap", "acc", "mae", "rmse", "bce", "mse", "loss"]}
            print(f"  -> 新最佳 val loss {best_loss:.4f}", flush=True)

    csv_file.close()

    # 用最佳模型在验证集上选阈值
    threshold, fbeta = select_threshold(best_val_probs, best_val_feas, beta=args.beta)
    model.load_state_dict(best_state)

    test_m = evaluate(model, loaders["test"], stats, device, lam=args.lam)
    best_metrics["threshold"] = threshold
    best_metrics["fbeta"] = fbeta
    best_metrics["test"] = {k: test_m[k] for k in ["ap", "acc", "mae", "rmse"]}

    json_path = save_checkpoint(model, stats, args, threshold, best_metrics, ckpt_path)

    print("\n" + "=" * 60)
    print(f"最佳阈值 (F{args.beta}) = {threshold:.4f}  (F={fbeta:.4f})")
    print(f"验证集: AP {best_metrics['ap']:.4f}  acc {best_metrics['acc']:.4f}  "
          f"MAE {best_metrics['mae']:.3f}s  RMSE {best_metrics['rmse']:.3f}s")
    print(f"测试集: AP {test_m['ap']:.4f}  acc {test_m['acc']:.4f}  "
          f"MAE {test_m['mae']:.3f}s  RMSE {test_m['rmse']:.3f}s")
    print(f"已保存: {ckpt_path} / {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
