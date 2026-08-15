"""TOF-Net 数据集生成器（模块一）。

多进程并行调用 gfold（内部 TOF 搜索模式）标注样本，带 rich TUI 实时显示
CPU 占用 / 可行率 / worker 并行数 / 滚动日志，支持断点续传与 JSONL 全量落盘。

用法示例（在项目根目录）：
    python -m generation.generate --n-samples 100000
    python -m generation.generate --pilot 2000       # 先导批次：测可行率，不写 shard
    python -m generation.generate --n-samples 200000 # 在已有 10 万基础上追加 10 万

追加数据：``--n-samples`` 表示「数据集总量目标」。若 data/manifest.json 已记录
N 条完成，再以更大的 --n-samples 运行，会自动从第 N 条继续，只补差额（同 seed
可整体复现；同 outdir/logdir 追加到同一数据集与日志）。
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from common import config as cfg
from generation import sampling

import gfold

try:
    import psutil
except ImportError:
    psutil = None

try:
    from rich.console import Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH = True
except ImportError:
    RICH = False


# --------------------------------------------------------------------------- #
# worker 端（必须顶层可 import，Windows spawn / PyInstaller 冻结进程需要）
# --------------------------------------------------------------------------- #

def build_config(d: dict):
    """用构造式注入建 gfold.Config（避开嵌套赋值静默失效的陷阱）。"""
    return gfold.Config(
        gfold.Spacecraft(
            wet_mass=d["wet_mass"],
            fuel=d["fuel"],
            real_max_thrust=d["real_max_thrust"],
            min_thrust_pct=d["min_thrust_pct"],
            max_thrust_pct=d["max_thrust_pct"],
            max_velocity=cfg.MAX_VELOCITY,
            initial_position=[d["x"], d["y"], d["z"]],
            initial_velocity=[d["vx"], d["vy"], d["vz"]],
            target_position=cfg.TARGET_POSITION,
            target_velocity=cfg.TARGET_VELOCITY,
            fuel_consumption=d["fuel_consumption"],
        ),
        gfold.Environment(
            gravity=cfg.GRAVITY,
            glide_slope_angle_deg=d["glide_slope_angle_deg"],
            max_angle_deg=d["max_angle_deg"],
        ),
        gfold.Solver(n=cfg.N, time_of_flight=None, tof_min=None, tof_max=None),
    )


def recover_tf(traj) -> float:
    """API 不暴露 time_of_flight，从 time_points 恢复：tf = n * dt（dt = tp[1]-tp[0]）。

    用「节点数 × 首段 dt」而非依赖最后一点，对上游 time_points 语义更稳健。
    """
    tp = np.asarray(traj.time_points)
    if len(tp) < 2:
        return 0.0
    return float(len(tp) * (tp[1] - tp[0]))


def worker(task):
    """求解一个样本，返回可序列化结果 dict。"""
    idx, seed = task
    rng = np.random.default_rng(seed)
    d = sampling.sample_one(rng)
    feats = sampling.feature_vector(d)

    result = {
        "index": idx,
        "features": feats,
        "feasible": False,
        "tf": None,
        "objective": None,
        "final_mass": None,
        "fuel_used": None,
        "status": "",
        "solve_ms": 0.0,
    }
    t0 = time.perf_counter()
    try:
        traj = gfold.solve(build_config(d))
        ms = (time.perf_counter() - t0) * 1000.0
        ok = traj.status in ("Solved", "AlmostSolved")
        result.update(
            feasible=ok,
            tf=recover_tf(traj),
            objective=float(traj.objective),
            final_mass=float(traj.final_mass),
            fuel_used=float(d["wet_mass"] - traj.final_mass),
            status=str(traj.status),
            solve_ms=ms,
        )
    except ValueError as e:
        result["status"] = str(e)
        result["solve_ms"] = (time.perf_counter() - t0) * 1000.0
    except Exception as e:  # noqa: BLE001
        result["status"] = f"{type(e).__name__}: {e}"
        result["solve_ms"] = (time.perf_counter() - t0) * 1000.0
    return result


# --------------------------------------------------------------------------- #
# TUI 渲染
# --------------------------------------------------------------------------- #

_BLOCKS = " ▁▂▃▄▅▆▇█"


def _core_char(p):
    idx = min(8, max(0, int(p / 100.0 * 8 + 0.5)))
    return _BLOCKS[idx]


def _cores_text(cores):
    per_line = 16
    return "\n".join(
        "".join(_core_char(c) for c in cores[i:i + per_line])
        for i in range(0, len(cores), per_line)
    )


def _categorize_status(status):
    if status in ("Solved", "AlmostSolved"):
        return status
    if "no feasible time-of-flight" in status:
        return "TOF-infeasible"
    if "PrimalInfeasible" in status:
        return "PrimalInfeasible"
    return "other"


def _short_status(status):
    if "no feasible time-of-flight" in status:
        return "no feasible TOF"
    if "PrimalInfeasible" in status:
        return "PrimalInfeasible"
    return status[:64]


def _fmt_log(res):
    idx = res["index"]
    if res["feasible"]:
        tag, style = "[OK  ]", "bold green"
        body = f"tf={res['tf']:.2f}s  mf={res['final_mass']:.0f}kg"
    else:
        tag, style = "[FAIL]", "bold red"
        body = _short_status(res["status"])
    return f"{tag} #{idx:<7d}  {body}  {res['solve_ms']:6.1f}ms", style


class Renderer:
    def __init__(self, total, max_workers):
        self.total = total
        self.max_workers = max_workers
        self._cpu_cores = None
        self._cpu_ts = 0.0

    def _sample_cpu(self):
        """节流采样：非阻塞 cpu_percent 测距上次调用的增量，高频调用会读到 0，
        这里缓存 0.5s，避免闪烁。返回 None 表示 psutil 不可用。"""
        if psutil is None:
            return None
        now = time.perf_counter()
        if self._cpu_cores is None or now - self._cpu_ts >= 0.5:
            self._cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
            self._cpu_ts = now
        return self._cpu_cores

    def cpu_panel_height(self):
        if psutil is None:
            return 3
        n = psutil.cpu_count() or 1
        return max(3, math.ceil(n / 16) + 2)

    def build(self, gen):
        st = gen.stats()
        elapsed = st["elapsed"]
        rate = st["done"] / elapsed if elapsed > 0 else 0.0
        remaining = self.total - st["done"]
        eta = remaining / rate if rate > 0 else 0.0

        header = Table.grid(expand=True)
        header.add_column(justify="left", ratio=2)
        header.add_column(justify="right", ratio=1)
        header.add_row(
            f"[bold cyan]TOF-Net Data Generation[/]  samples "
            f"{st['done']:,}/{self.total:,}",
            f"elapsed {elapsed:>7.1f}s  eta {eta:>6.1f}s",
        )
        header.add_row(
            f"[bold]feasible[/] {st['n_feas']:,} ({st['feas_rate']*100:5.1f}%)   "
            f"[bold]rate[/] {rate:6.1f}/s   [bold]shards[/] {st['n_shards']}",
            f"[bold]workers[/] {st['active']}/{self.max_workers}",
        )

        cpu_parts = []
        cores = self._sample_cpu()
        if cores is not None:
            total_cpu = float(np.mean(cores))
            cpu_parts.append(
                Text(f"CPU [{total_cpu:4.1f}%]\n", style="bold")
                + Text(_cores_text(cores), style="green")
            )
        else:
            cpu_parts.append(Text("CPU n/a (psutil 不可用)"))

        log_text = Text()
        for res in gen.log_buffer:
            line, style = _fmt_log(res)
            log_text.append(line + "\n", style=style)

        layout = Layout()
        layout.split_column(
            Layout(Panel(header), name="header", size=5),
            Layout(Panel(Group(*cpu_parts), title="CPU"),
                   name="cpu", size=self.cpu_panel_height()),
            Layout(Panel(log_text, title="workers"), name="log"),
        )
        return layout


# --------------------------------------------------------------------------- #
# 生成器主类
# --------------------------------------------------------------------------- #

class Generator:
    def __init__(self, args):
        self.args = args
        self.total = args.n_samples
        self.max_workers = args.workers
        self.base_seed = args.seed
        self.shard_size = args.shard_size
        self.write_shards = not args.no_write

        self.outdir = args.outdir
        self.logdir = args.logdir
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)

        self.start_time = time.perf_counter()
        self.n_done = 0
        self.n_feas = 0
        self.status_counter = Counter()
        self.n_shards = 0

        self.log_buffer = deque(maxlen=40)
        self._shard_buf = []

        self.log_path = os.path.join(self.logdir, args.logfile)
        self.manifest_path = os.path.join(self.outdir, "manifest.json")
        self.meta_path = os.path.join(self.outdir, "meta.json")

        self._load_or_init_manifest()
        self._write_meta()
        self._log_file = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115

    # -- 状态 -------------------------------------------------------------- #
    def stats(self):
        elapsed = time.perf_counter() - self.start_time
        return {
            "done": self.n_done,
            "n_feas": self.n_feas,
            "feas_rate": self.n_feas / self.n_done if self.n_done else 0.0,
            "elapsed": elapsed,
            "active": min(self.n_pending, self.max_workers),
            "n_shards": self.n_shards,
        }

    # -- manifest / meta ---------------------------------------------------- #
    def _load_or_init_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            self.start_idx = m.get("n_done", 0)
            self._shard_list = m.get("shards", [])
            self.n_shards = len(self._shard_list)
        else:
            self.start_idx = 0
            self._shard_list = []
            self.n_shards = 0

    def _write_meta(self):
        try:
            from importlib.metadata import version
            gfold_version = version("gfold")
        except Exception:  # noqa: BLE001
            gfold_version = "unknown"
        meta = {
            "schema_version": cfg.SCHEMA_VERSION,
            "features": cfg.FEATURES,
            "norm_bounds": {k: list(v) for k, v in cfg.NORM_BOUNDS.items()},
            "ranges": {k: list(v) for k, v in cfg.RANGES.items()},
            "g": cfg.KERBIN_G,
            "n": cfg.N,
            "max_velocity": cfg.MAX_VELOCITY,
            "gravity": cfg.GRAVITY,
            "target_position": cfg.TARGET_POSITION,
            "target_velocity": cfg.TARGET_VELOCITY,
            "gfold_version": gfold_version,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _save_manifest(self):
        with open(self.manifest_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"n_done": self.start_idx + self.n_done,
                       "shards": self._shard_list}, f)
        os.replace(self.manifest_path + ".tmp", self.manifest_path)

    # -- 结果处理 ---------------------------------------------------------- #
    def _process_result(self, res):
        self.n_done += 1
        if res["feasible"]:
            self.n_feas += 1
        self.status_counter[_categorize_status(res["status"])] += 1
        self.log_buffer.append(res)

        self._log_file.write(json.dumps(res) + "\n")

        if self.write_shards:
            self._shard_buf.append(res)
            if len(self._shard_buf) >= self.shard_size:
                self._flush_shard()

    def _flush_shard(self):
        if not self._shard_buf:
            return
        start = self.start_idx + self.n_done - len(self._shard_buf)
        path = os.path.join(self.outdir, f"shard_{start:08d}.npz")
        tmp = path + ".tmp.npz"   # 以 .npz 结尾，numpy 不会追加后缀

        m = len(self._shard_buf)
        X = np.empty((m, cfg.N_FEATURES), dtype=np.float64)
        y_feas = np.zeros(m, dtype=np.int8)
        y_tf = np.full(m, np.nan, dtype=np.float64)
        objective = np.full(m, np.nan, dtype=np.float64)
        final_mass = np.full(m, np.nan, dtype=np.float64)
        fuel_used = np.full(m, np.nan, dtype=np.float64)
        solve_ms = np.zeros(m, dtype=np.float64)
        index = np.zeros(m, dtype=np.int64)

        for i, r in enumerate(self._shard_buf):
            X[i] = r["features"]
            y_feas[i] = 1 if r["feasible"] else 0
            index[i] = r["index"]
            solve_ms[i] = r["solve_ms"]
            if r["feasible"]:
                y_tf[i] = r["tf"]
                objective[i] = r["objective"]
                final_mass[i] = r["final_mass"]
                fuel_used[i] = r["fuel_used"]

        np.savez_compressed(
            tmp,
            X=X, y_feasible=y_feas, y_tf=y_tf,
            objective=objective, final_mass=final_mass,
            fuel_used=fuel_used, solve_ms=solve_ms, index=index,
        )
        os.replace(tmp, path)

        self._shard_list.append(os.path.basename(path))
        self.n_shards += 1
        self._shard_buf = []
        self._log_file.flush()
        self._save_manifest()

    # -- 主循环 ------------------------------------------------------------ #
    def run(self):
        if self.start_idx >= self.total:
            print(f"[resume] 已有 {self.start_idx:,} 条 >= 目标 {self.total:,}，"
                  f"无需生成。若想追加，请调大 --n-samples。", flush=True)
            self._log_file.close()
            return
        if self.start_idx > 0:
            print(f"[resume] 检测到已完成 {self.start_idx:,} 条，"
                  f"本次将补生成 {self.total - self.start_idx:,} 条至总量 "
                  f"{self.total:,}（同 seed 可整体复现）。", flush=True)

        ex = ProcessPoolExecutor(max_workers=self.max_workers)
        pending = {}
        next_idx = self.start_idx
        self.n_pending = 0

        def submit_more():
            nonlocal next_idx
            while len(pending) < self.max_workers * 4 and next_idx < self.total:
                fut = ex.submit(worker, (next_idx, self.base_seed + next_idx))
                pending[fut] = next_idx
                next_idx += 1
            self.n_pending = len(pending)

        is_tty = sys.stdout.isatty() and not self.args.no_tui and RICH
        renderer = Renderer(self.total, self.max_workers) if is_tty else None

        try:
            if renderer is not None:
                with Live(renderer.build(self), refresh_per_second=8,
                          screen=True) as live:
                    submit_more()
                    while pending:
                        done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                        for fut in done:
                            pending.pop(fut)
                            self._process_result(fut.result())
                        submit_more()
                        live.update(renderer.build(self))
            else:
                last_log = 0.0
                submit_more()
                while pending:
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending.pop(fut)
                        self._process_result(fut.result())
                    submit_more()
                    now = time.perf_counter()
                    if now - last_log > 5.0:
                        st = self.stats()
                        rate = st["done"] / st["elapsed"] if st["elapsed"] else 0.0
                        print(f"[{now - self.start_time:7.1f}s] "
                              f"done={st['done']:,}/{self.total:,} "
                              f"feasible={st['feas_rate']*100:5.1f}% "
                              f"rate={rate:5.1f}/s "
                              f"workers={st['active']}/{self.max_workers}",
                              flush=True)
                        last_log = now
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
            if self.write_shards:
                self._flush_shard()
            self._log_file.flush()
            self._log_file.close()
            self._save_manifest()

        self._print_summary()

    def _print_summary(self):
        st = self.stats()
        total_now = self.start_idx + self.n_done
        print("\n" + "=" * 60)
        print(f"生成完成：本次 {st['done']:,} 条（数据集总量 {total_now:,} 条）")
        print(f"  可行：{st['n_feas']:,} ({st['feas_rate']*100:.1f}%)  "
              f"不可行：{st['done'] - st['n_feas']:,}")
        print(f"  耗时：{st['elapsed']:.1f}s  速率：{st['done']/st['elapsed']:.1f}/s")
        if self.status_counter:
            print("  状态分布：")
            for status, cnt in self.status_counter.most_common():
                print(f"    {cnt:>8,}  {status}")
        print("=" * 60)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="TOF-Net 数据集生成")
    p.add_argument("--n-samples", type=int, default=cfg.DEFAULT_N_SAMPLES,
                   help="数据集总量目标（存在 manifest 时自动续传至该总量）")
    p.add_argument("--pilot", type=int, default=0,
                   help="先导批次：仅跑 N 个样本测可行率，写入 *_pilot 目录")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    p.add_argument("--shard-size", type=int, default=cfg.DEFAULT_SHARD_SIZE)
    p.add_argument("--outdir", default="data")
    p.add_argument("--logdir", default="logs")
    p.add_argument("--logfile", default="samples.jsonl")
    p.add_argument("--no-write", action="store_true", help="不写 shard（校准用）")
    p.add_argument("--no-tui", action="store_true", help="禁用 TUI")
    return p.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args()
    if args.pilot > 0:
        args.n_samples = args.pilot
        args.no_write = True
        args.outdir = args.outdir + "_pilot"
        args.logfile = "pilot.jsonl"
    gen = Generator(args)
    gen.run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
