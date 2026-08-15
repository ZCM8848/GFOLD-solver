"""TOF-Net 推理：加载 checkpoint → 给定状态输出 (p_feasible, tf)。

模型常驻内存（只加载一次），输入为 14 维原始物理量，内部做同样的
min-max 归一化 + 反标准化，返回可行概率与 TF 秒数。

用法：
    from inference.predictor import TOFNetPredictor
    pred = TOFNetPredictor("models/tofnet.pt")
    p, tf = pred.predict([x, y, z, vx, vy, vz, dry_mass, fuel, real_max_thrust,
                          min_thrust_pct, max_thrust_pct, fuel_consumption,
                          glide_slope_angle_deg, max_angle_deg])
    feas, p, tf = pred.decide(features)   # 按训练时选定的阈值判断
"""

import torch

from common.model import TOFNet


class TOFNetPredictor:
    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.features = ckpt["features"]
        self.norm_bounds = ckpt["norm_bounds"]
        self.tf_mean = float(ckpt["tf_mean"])
        self.tf_std = float(ckpt["tf_std"])
        self.threshold = float(ckpt["threshold"])

        self.model = TOFNet(
            in_features=ckpt["in_features"],
            hidden=tuple(ckpt["hidden"]),
            activation=ckpt["activation"],
            dropout=ckpt["dropout"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        lo = [self.norm_bounds[f][0] for f in self.features]
        hi = [self.norm_bounds[f][1] for f in self.features]
        self._lo = torch.tensor(lo, dtype=torch.float32, device=self.device)
        self._hi = torch.tensor(hi, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, features):
        """features: (14,) 或 (B,14) 原始特征（FEATURES 顺序）。

        单样本返回 (float, float)；批量返回 (tensor(B,), tensor(B,))。
        """
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        xn = (x - self._lo) / (self._hi - self._lo)
        logits, tf_std = self.model(xn)
        p = torch.sigmoid(logits).squeeze(-1)
        tf = tf_std.squeeze(-1) * self.tf_std + self.tf_mean
        if single:
            return float(p[0]), float(tf[0])
        return p, tf

    def predict_from_dict(self, d):
        """d: 以特征名为键的 dict（键齐全），返回 (p, tf)。"""
        return self.predict([d[f] for f in self.features])

    def decide(self, features, threshold=None):
        """返回 (feasible_bool, p, tf)。threshold 默认用训练时选定的值。"""
        p, tf = self.predict(features)
        th = self.threshold if threshold is None else threshold
        return p >= th, p, tf


def main():
    import argparse
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="TOF-Net 推理 demo")
    ap.add_argument("--ckpt", default="models/tofnet.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pred = TOFNetPredictor(args.ckpt, device=args.device)
    print(f"[load] {args.ckpt}  features={len(pred.features)}  "
          f"threshold={pred.threshold:.4f}  device={pred.device}")

    # 用一个接近默认 gfold 场景的状态做演示
    sample = [450.0, -330.0, 2400.0, -40.0, 10.0, -10.0,
              16000.0, 40000.0, 1_500_000.0, 0.4, 0.98,
              0.00035, 45.0, 10.0]
    p, tf = pred.predict(sample)
    feas, _, _ = pred.decide(sample)
    print(f"[demo] p_feasible={p:.4f}  tf={tf:.2f}s  "
          f"决策(threshold {pred.threshold:.4f})={feas}")


if __name__ == "__main__":
    main()
