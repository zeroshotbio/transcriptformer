# src/transcriptformer/model/masks.py
import torch


# -----------------------------------------------------------------------
# Pad-mask factory
# -----------------------------------------------------------------------
def pad_mask_factory(mask: torch.Tensor | None):
    """
    Returns a function `pad_mask(b, h, q_idx, kv_idx) -> torch.bool`.

    • If `mask` is None ➜ every position is valid  (True)
    • Otherwise        ➜ rows are batch, columns are key/val positions
      – Out-of-range kv_idx (>= mask.shape[1]) are treated as padded (=False)
    """
    if mask is None:
        # trivial mask that always allows attention
        return lambda b, h, q_idx, kv_idx: torch.tensor(True, device=q_idx.device)

    B, M = mask.shape  # M == # tokens covered by the mask
    M_minus_1 = M - 1  # handy constant for clamping

    # make a tensor constant so the whole body stays on-device
    M_const = torch.tensor(M, device=mask.device)

    def pad_mask(b, h, q_idx, kv_idx):
        # kv_idx  is a 0-D tensor coming from functorch / FlexAttention
        valid = kv_idx < M_const  # bool tensor
        safe_k = torch.where(valid, kv_idx, M_minus_1)  # clamp when out-of-range
        return valid & mask[b, safe_k]  # final boolean

    return pad_mask


# -----------------------------------------------------------------------
# Causal-mask factory
# -----------------------------------------------------------------------
def causal_mask_factory(start_row: int = 0, start_col: int = 0, offset: int = 0):
    """Standard causal / autoregressive mask with optional window offset."""

    def causal_mask(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx + offset) & (q_idx >= start_row) & (kv_idx >= start_col)

    return causal_mask
