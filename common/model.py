"""双头 MLP：共享 backbone + 可行性分类头 + TOF 回归头。

训练与推理共用本定义，保证架构一致。forward 返回：
    logits (B,1)  可行性原始 logit（训练用 BCEWithLogitsLoss，推理时 sigmoid 得概率）
    tf_pred (B,1) TOF 预测（标准化空间，推理时反标准化回秒）
"""

import torch.nn as nn

_ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class TOFNet(nn.Module):
    def __init__(self, in_features=14, hidden=(128, 128, 64),
                 dropout=0.0, activation="silu"):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        layers = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.class_head = nn.Linear(prev, 1)   # 可行性 logit
        self.reg_head = nn.Linear(prev, 1)     # 标准化 TF

    def forward(self, x):
        h = self.backbone(x)
        return self.class_head(h), self.reg_head(h)
