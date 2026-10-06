"""采样策略：greedy / top-k / top-p（nucleus）/ temperature（M5）。"""
import torch
import torch.nn.functional as F


def sample_token(logits, top_k: int = 0, top_p: float = 1.0,
                 temperature: float = 1.0):
    """logits: (..., V) -> 采样 id（同形状去掉最后一维）。

    - temperature<=0 退化为 greedy，避免除零
    - top_k>0 只保留概率最高的 k 个
    - top_p<1 按概率降序累加到阈值截断（nucleus），截断后重新归一化
    """
    logits = logits.clone()
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / float(temperature)

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth,
                             torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # 累计概率首次超过 top_p 之后的位置全部屏蔽（至少保留第 1 个）
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(
            -1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]),
                              num_samples=1).reshape(probs.shape[:-1])
