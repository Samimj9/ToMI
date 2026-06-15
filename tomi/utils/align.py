import torch

def align_sequence(a: torch.Tensor, b: torch.Tensor):
    """
    Align two activation tensors along sequence dimension.
    Supports (B, S, D) and similar transformer formats.
    """

    if a.dim() >= 3 and b.dim() >= 3:
        min_len = min(a.shape[1], b.shape[1])
        return a[:, :min_len], b[:, :min_len]

    # fallback for flat tensors
    min_len = min(a.numel(), b.numel())
    return a.flatten()[:min_len], b.flatten()[:min_len]