"""训练曲线可视化：读 logs/train_metrics.csv 画收敛过程。

用法：
    python -m visualization.training --csv logs/train_metrics.csv
"""

import argparse
import csv
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization._common import save


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="训练曲线")
    ap.add_argument("--csv", default="logs/train_metrics.csv")
    args = ap.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    ep = [r["epoch"] for r in rows]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Loss", "AP / accuracy", "TF MAE (s)", "TF RMSE (s)"])

    fig.add_trace(go.Scatter(x=ep, y=[r["train_loss"] for r in rows],
                             name="train loss"), 1, 1)
    fig.add_trace(go.Scatter(x=ep, y=[r["val_loss"] for r in rows],
                             name="val loss"), 1, 1)

    fig.add_trace(go.Scatter(x=ep, y=[r["val_ap"] for r in rows],
                             name="val PR-AUC"), 1, 2)
    fig.add_trace(go.Scatter(x=ep, y=[r["val_acc"] for r in rows],
                             name="val accuracy"), 1, 2)

    fig.add_trace(go.Scatter(x=ep, y=[r["val_mae"] for r in rows],
                             name="MAE"), 2, 1)
    fig.add_trace(go.Scatter(x=ep, y=[r["val_rmse"] for r in rows],
                             name="RMSE"), 2, 2)

    fig.update_layout(height=700, title="训练收敛曲线", legend=dict(orientation="h"))
    save(fig, "training_curves.html")


if __name__ == "__main__":
    main()
