"""共享契约包：特征 schema、采样范围、归一化边界、网络架构（三段共用）。

注意：`config.py` 不依赖 torch（纯 Python + numpy），数据生成器打包 exe 时
只 import config，不会拉进 torch；`model.py` 依赖 torch，仅供训练/推理 import。
"""
