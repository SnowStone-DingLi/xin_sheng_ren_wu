"""旋转位置编码 RoPE（相邻两维一组的实现）。"""
import torch


def build_rope_cache(seq_len: int, head_dim: int, base: int = 10000,
                     device=None, dtype=torch.float32):
    """返回 cos/sin 缓存，形状 (seq_len, head_dim)。"""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device,
                                            dtype=dtype) / half))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    ang = torch.outer(positions, inv_freq)          # (T, half)
    cos = torch.repeat_interleave(ang.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(ang.sin(), 2, dim=-1)
    return cos, sin


def apply_rope(x, cos, sin, offset: int = 0):
    """对 Q/K 施加 RoPE。x: (B,H,T,D)；cos/sin: (max_len,D)。"""
    T = x.size(-2)
    cos = cos[offset:offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
    rot2 = x1 * sin[..., 0::2] + x2 * cos[..., 0::2]
    out = torch.stack((rot1, rot2), dim=-1).flatten(-2)
    return out.to(x.dtype)
