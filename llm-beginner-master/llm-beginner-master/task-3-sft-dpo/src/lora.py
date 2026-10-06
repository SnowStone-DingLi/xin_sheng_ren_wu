"""手写 LoRA：低秩矩阵注入、forward 叠加、merge 还原（M1）。

- 不依赖 peft
- A: (in, r) kaiming 初始化；B: (r, out) 零初始化 -> 训练开始不改变原输出
- scaling = alpha / r
- 冻结原权重 W，只训练 A/B
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features

        self.lora_A = nn.Parameter(torch.empty(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # 注入可能发生在 model.to(device) 之后，新参数需跟随基座设备/精度
        self.lora_A.data = self.lora_A.data.to(base.weight.device, base.weight.dtype)
        self.lora_B.data = self.lora_B.data.to(base.weight.device, base.weight.dtype)
        self.lora_dropout = nn.Dropout(dropout)

        # 原权重冻结；bias 也不训练
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            delta = self.lora_dropout(x) @ self.lora_A @ self.lora_B
            out = out + delta * self.scaling
        return out

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            delta = (self.lora_A @ self.lora_B).t() * self.scaling
            self.base.weight.add_(delta)
            self.merged = True


def _replace_linear(parent, name, child, r, alpha, dropout):
    modules = dict(parent.named_modules())
    replaced = 0
    for mod_name, mod in list(parent.named_modules()):
        for child_name, layer in list(mod.named_children()):
            if isinstance(layer, nn.Linear) and child_name in name:
                wrapped = LoRALinear(layer, r=r, alpha=alpha, dropout=dropout)
                setattr(mod, child_name, wrapped)
                replaced += 1
    return replaced


def inject_lora(model, target_modules=("q_proj", "v_proj"),
                r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
    """把模型中名为 target_modules 的 nn.Linear 原地替换为 LoRALinear。

    同时冻结全部参数，再只放开 LoRA 的 A/B。
    """
    target_modules = tuple(target_modules)
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for mod in list(model.modules()):
        for child_name, layer in list(mod.named_children()):
            if isinstance(layer, nn.Linear) and child_name in target_modules:
                setattr(mod, child_name,
                        LoRALinear(layer, r=r, alpha=alpha, dropout=dropout))
                replaced += 1

    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)

    model._lora_targets = list(target_modules)
    model._lora_r = r
    model._lora_alpha = alpha
    print(f"[LoRA] 注入 {replaced} 个线性层 targets={target_modules} "
          f"r={r} alpha={alpha}")
    return model


@torch.no_grad()
def merge_lora(model):
    """把 LoRA 增量合回 base 权重并清理分支；合并后前向与训练时一致。"""
    n = 0
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.merge()
            n += 1
    print(f"[LoRA] 已合并 {n} 个层")
    return model


def lora_state_dict(model):
    """只导出 LoRA 参数，便于保存 adapter。"""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            out[f"{name}.lora_A"] = m.lora_A.detach().cpu()
            out[f"{name}.lora_B"] = m.lora_B.detach().cpu()
    return out


def load_lora_state_dict(model, state):
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.lora_A.copy_(state[f"{name}.lora_A"])
                m.lora_B.copy_(state[f"{name}.lora_B"])
    return model


ADAPTER_FILENAME = "adapter.pt"
CONFIG_FILENAME = "adapter_config.json"


def save_adapter(model, out_dir):
    """保存 LoRA adapter（只含 A/B）+ 配置。"""
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    torch.save(state, out_dir / ADAPTER_FILENAME)
    cfg = {"r": getattr(model, "_lora_r", 8),
           "alpha": getattr(model, "_lora_alpha", 16),
           "target_modules": getattr(model, "_lora_targets",
                                     ["q_proj", "v_proj"]),
           "n_adapter_tensors": len(state)}
    (out_dir / CONFIG_FILENAME).write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LoRA] adapter 已保存到 {out_dir}（{len(state)} 个张量）")


def build_lora_model(model_path, adapter_dir=None, merge=False,
                     dtype=None, device=None):
    """加载基座并注入 LoRA；可选加载 adapter 权重或直接 merge。"""
    from transformers import AutoModelForCausalLM
    import torch
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
    if adapter_dir is not None:
        from pathlib import Path
        state = torch.load(Path(adapter_dir) / ADAPTER_FILENAME, map_location="cpu")
        load_lora_state_dict(model, state)
        if merge:
            merge_lora(model)
    if device is not None:
        model.to(device)
    return model
