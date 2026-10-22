"""Differentiable float64 evidence losses. No clipping or alpha rescaling."""

import torch
import torch.nn.functional as F


def alpha_from_logits(logits, prior_raw, mapping="exp"):
    if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite [B,K], K>=2")
    # 先转 float64，再计算 exp / Gamma，保留梯度与证据幅度。
    z = logits.double()
    raw = torch.as_tensor(prior_raw, device=z.device, dtype=torch.float64)
    if raw.numel() != 1 or not torch.isfinite(raw).all():
        raise ValueError("prior_raw must be one finite scalar (log prior)")
    if mapping not in ("exp", "softplus"):
        raise ValueError("unknown evidence mapping")
    prior = raw.exp()
    evidence = z.exp() if mapping == "exp" else F.softplus(z)
    alpha = evidence + prior
    validate_alpha(alpha)
    return alpha


def validate_alpha(alpha):
    if alpha.ndim != 2 or alpha.numel() == 0:
        raise ValueError("alpha must be nonempty [B,K]")
    if not torch.isfinite(alpha).all() or not (alpha > 0).all():
        raise FloatingPointError("nonpositive or nonfinite alpha; refusing to clip/rescale")
    if not torch.isfinite(alpha.sum(-1)).all():
        raise FloatingPointError("Dirichlet concentration overflow")


def dirichlet_kl(teacher_alpha, student_alpha):
    """KL(Dir(teacher)||Dir(student)), per sample, with teacher detached."""
    a, b = teacher_alpha.detach().double(), student_alpha.double()
    validate_alpha(a)
    validate_alpha(b)
    if a.shape != b.shape:
        raise ValueError("teacher/student alpha shapes differ")
    sa, sb = a.sum(-1), b.sum(-1)
    value = (
        torch.lgamma(sa)
        - torch.lgamma(a).sum(-1)
        - torch.lgamma(sb)
        + torch.lgamma(b).sum(-1)
        + ((a - b) * (torch.digamma(a) - torch.digamma(sa)[:, None])).sum(-1)
    )
    if not torch.isfinite(value).all():
        raise FloatingPointError("nonfinite Dirichlet KL")
    if (value < -1e-7).any():
        raise FloatingPointError(f"negative Dirichlet KL beyond tolerance: {value.min().item()}")
    return torch.where(value < 0, value * 0, value)


def evidential_ce(alpha, labels):
    """真实标签监督：E_Dir[-log p_y] = digamma(alpha0)-digamma(alpha_y)。"""
    validate_alpha(alpha)
    if labels.dtype != torch.long or labels.shape != (len(alpha),):
        raise ValueError("labels must be int64 [B]")
    if not ((labels >= 0) & (labels < alpha.shape[1])).all():
        raise ValueError("labels outside class range")
    return (
        torch.digamma(alpha.sum(-1)) - torch.digamma(alpha.gather(1, labels[:, None]).squeeze(1))
    ).mean()
