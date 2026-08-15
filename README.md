# TOF-Net

为 [G-FOLD](https://github.com/samutoljamo/g-fold) 求解器外挂一个**学习型飞行时间（TOF）预测器**：离线蒙特卡洛采样生成「状态/约束 → 最优 TOF」数据集，训练一个双头 MLP（可行性分类 + TOF 回归），在线时跳过求解器内部的 TOF 搜索，实现高频 MPC 动力下降制导。

本仓库实现三模块：**数据生成**、**训练**、**推理**。

## 目录结构

```
GFOLD-solver/
├── common/                 # 共享契约（三段共用）
│   ├── config.py           # 特征 schema + 采样范围 + min-max 归一化边界 + 常量（无 torch 依赖）
│   └── model.py            # 双头 MLP 架构（训练/推理共用，依赖 torch）
├── generation/             # 模块一：数据生成
│   ├── sampling.py         # 因果一致采样（纯函数）
│   └── generate.py         # 多进程生成 + rich TUI + shard 续传 + JSONL 落盘
├── training/               # 模块二：训练
│   ├── dataset.py          # 读 shard + 归一化 + 分层切分（支持 GPU 显存预加载）
│   └── train.py            # BCE+λ·MSE 训练 + PR 阈值选择 + checkpoint
├── inference/              # 模块三：推理
│   └── predictor.py        # 加载 checkpoint → (p_feasible, tf)
├── scripts/
│   ├── run_generate.py     # PyInstaller 打包入口
│   └── build_exe.ps1       # 打包脚本
├── docs/                   # 设计文档
│   ├── TOF-Net设计方案.md
│   └── gfold_Python_API文档.md
├── requirements.txt
└── .gitignore
```

## 安装

环境：miniforge 环境 `KRPC`（Python 3.12）。

```bash
# 核心依赖
python -m pip install gfold numpy rich psutil
# torch 用 CPU 索引单独装（训练/推理用，数据生成 exe 不需要）
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 约定（三段共享的契约，勿改动）

### 坐标系

- **z-up**；重力沿 −z，大小固定为 Kerbin 表面重力 `g = 9.81 m/s²`（`GRAVITY = [0,0,-9.81]`）
- 目标着陆点为**原点**，目标速度 `[0,0,0]`
- 位置、速度均为相对目标点的量
- 离散节点数固定 `n = 100`；`max_velocity = 600`（保证初始速度不违反 SOC 约束）

### 特征 schema（14 维，顺序即契约）

```
[x, y, z, vx, vy, vz, dry_mass, fuel, real_max_thrust,
 min_thrust_pct, max_thrust_pct, fuel_consumption,
 glide_slope_angle_deg, max_angle_deg]
```

### 归一化

min-max 到 `[0,1]`，边界在 `common/config.py` 的 `NORM_BOUNDS`（**先验已知**，非统计量），
生成时写入 `meta.json`，训练/推理读同一份，保证三段一致。

### 因果采样链（保证物理自洽）

```
质量链：dry_mass -> dry_frac(=dry/wet_full) -> wet_full
        -> remaining_frac -> fuel, wet_mass
推力链：twr_max -> real_max_thrust = twr_max * wet_mass * g
        （min_thrust_pct < max_thrust_pct 恒成立；有效 TWR = twr_max * max_pct > 1）
Isp   -> fuel_consumption = 1 / (Isp * 9.80665)
```

完整范围见 `common/config.py` 的 `RANGES`。

### gfold API 两个关键事实（详见 docs/gfold_Python_API文档.md）

1. **嵌套字段赋值静默失效**：必须用构造式注入或整体替换嵌套对象，不能
   `cfg.solver.time_of_flight = ...`。
2. **最优 TF 不直接暴露**：从结果恢复 `tf = time_points[-1] * n/(n-1)`。

## 数据集格式

生成产物全部在 `data/`（shard + 元数据）与 `logs/`（逐样本 JSONL）。

### shard：`data/shard_<start>.npz`

每个 shard 默认 10000 条（`--shard-size` 可调），字段：

| 数组 | 形状 | dtype | 说明 |
|---|---|---|---|
| `X` | (m, 14) | float64 | 原始特征（未归一化），列顺序见 schema |
| `y_feasible` | (m,) | int8 | 可行性标签 0/1 |
| `y_tf` | (m,) | float64 | 最优 TOF（秒）；不可行样本为 NaN |
| `objective` | (m,) | float64 | 优化目标 = ln(末端质量)；不可行为 NaN |
| `final_mass` | (m,) | float64 | 末端质量 kg；不可行为 NaN |
| `fuel_used` | (m,) | float64 | 燃料消耗 kg；不可行为 NaN |
| `solve_ms` | (m,) | float64 | 求解耗时 ms |
| `index` | (m,) | int64 | 全局样本索引（同 seed 可复现；shard 内乱序，以 index 为准） |

### `data/meta.json` —— 契约快照

记录 `schema_version`、`features`（顺序）、`norm_bounds`、`ranges`、`g`、`n` 等。
训练与推理必须读取此文件而非硬编码。

### `data/manifest.json` —— 续传状态

```json
{"n_done": 100000, "shards": ["shard_00000000.npz", ...]}
```

### `logs/samples.jsonl` —— 逐样本全量日志（provenance）

每行一个 JSON：`index`、`features`、`feasible`、`tf`、`objective`、`final_mass`、
`fuel_used`、`status`、`solve_ms`。不可行样本的 `status` 为 gfold 异常消息
（多为 `infeasible: no feasible time-of-flight in [...]`）。

## 用法（模块一）

在项目根目录运行：

```bash
# 生成 10 万样本（默认 32 核 = CPU 逻辑核数、seed=0、shard=1 万）
python -m generation.generate --n-samples 100000

# 先导批次：跑 2000 条测可行率（写入 data_pilot/，不污染正式数据）
python -m generation.generate --pilot 2000

# 追加数据：在已有 10 万基础上再补 10 万（同 seed/outdir 自动续传）
python -m generation.generate --n-samples 200000

# 指定核数 / 种子 / shard 大小 / 输出目录
python -m generation.generate --n-samples 100000 --workers 64 --seed 1 --shard-size 20000
```

### 全部参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n-samples` | 100000 | 数据集**总量目标**；存在 manifest 时自动续传补差额 |
| `--pilot N` | 0 | 先导批次：只跑 N 条测可行率，写入 `*_pilot` 目录 |
| `--workers` | `os.cpu_count()` | 并行进程数（gfold 求解不释放 GIL，需多进程） |
| `--seed` | 0 | 样本 i 用 `seed+i` 生成，同 seed 可整体复现 |
| `--shard-size` | 10000 | 每个 .npz 的样本数 |
| `--outdir` | data | shard + manifest + meta 输出目录 |
| `--logdir` | logs | JSONL 日志目录 |
| `--logfile` | samples.jsonl | 日志文件名 |
| `--no-write` | false | 不写 shard（校准用） |
| `--no-tui` | false | 禁用 TUI |

### 追加 / 续传语义（回答「先 10 万、再补 10 万」）

- `--n-samples` 始终表示**数据集总量**。首次 `--n-samples 100000` 生成 0~99999；
  再运行 `--n-samples 200000`（同 `--outdir`/`--logdir`/`--seed`）会检测 manifest
  里已完成 100000 条，自动从第 100000 条继续，只补差额，追加到同一批 shard 与日志。
- 用**相同 `--seed`** 时，续传不影响整体可复现性（每个索引的采样种子确定）。
- 崩溃安全：shard 先写 `.tmp.npz` 再原子改名，manifest 仅在 shard 落盘后更新；
  中断后重跑即从最后完成的 shard 边界续传。

### TUI

交互式终端下自动开启 rich 全屏 TUI：顶部进度（总量/可行率/速率/shard/耗时 ETA）、
中间逐核 CPU 块状图 + 并行 worker 数、底部滚动 worker 成功/失败日志。
非 TTY（服务器 SSH）自动降级为每 5 秒一行进度；`--no-tui` 强制关闭。
`Ctrl+C` 优雅退出：冲刷当前 shard + 写 manifest，支持续传。

## 打包 exe（丢给服务器）

数据生成器不依赖 torch，exe 只含 gfold/numpy/rich/psutil，体积小、启动快。

```powershell
# PowerShell（KRPC 环境）
powershell -File scripts\build_exe.ps1
# 或手动：
pyinstaller --name tofnet_generate --onedir --clean --noconfirm `
    --paths . --collect-all gfold --hidden-import generation.generate `
    scripts\run_generate.py
```

产物在 `dist/tofnet_generate/`，**整个文件夹**拷贝到服务器后：

```bash
./tofnet_generate.exe --n-samples 100000 --workers <服务器核数>
```

- 用 `--onedir`（非 onefile）：多进程 spawn 时子进程直接加载已解压的二进制，避免 onefile 每进程重复解压的开销。
- 服务器上无需 Python/gfold/torch 环境，参数全部通过命令行传入，保持可调。

## 训练（模块二）

```bash
# 全量训练（有 GPU 自动用 GPU 并预加载数据进显存）
python -m training.train

# 常用参数
python -m training.train --epochs 50 --batch 4096 --lr 3e-4 --device auto
```

- 损失 `BCE(可行性) + λ·MSE(TF, 仅可行样本)`；TF 用 z-score、输入用 min-max（同 `meta.json`）
- 按 feasible 分层切 80/10/10；验证集 PR 曲线选阈值（`--beta` 控 Fβ，β<1 偏 precision）
- 产物 `models/tofnet.pt`（权重 + 契约自包含）+ `models/tofnet.json`（人类可读）

## 推理（模块三）

```python
from inference.predictor import TOFNetPredictor
pred = TOFNetPredictor("models/tofnet.pt")     # 常驻内存，只加载一次
p, tf = pred.predict(features)                  # features: 14 维原始物理量（FEATURES 顺序）
feasible, p, tf = pred.decide(features)         # 按训练阈值决策
```

```bash
python -m inference.predictor --ckpt models/tofnet.pt   # demo
```

- 单样本返回 `(float, float)`；批量 `(B,14)` 返回张量；`predict_from_dict(dict)` 按特征名取
- 在线用 `decide`：`p >= threshold` 判可行；低于阈值应 fallback 到完整 TOF 搜索

## 评估（下游价值验证）

```bash
python -m training.evaluate --n-feasible 1000 --n-infeasible 1000
```

对 held-out 可行样本用预测 TF 固定重解，报告求解成功率 + 燃料 suboptimality，
并与「常数 TF」基线、oracle（最优 TF 标签）对照；同时报告阈值处的分类代价。

## 待实现

- 在线 MPC 闭环（kRPC + KSP）——属 KSP-Auto-Landing 项目职责，本仓库只提供估计器
