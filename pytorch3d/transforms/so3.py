import torch


def so3_exp_map(log_rot):
    theta = torch.linalg.vector_norm(log_rot, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = log_rot / theta
    a = torch.sin(theta)[..., None]
    b = (1.0 - torch.cos(theta))[..., None]
    x, y, z = axis.unbind(-1)
    k = torch.stack((torch.zeros_like(x), -z, y, z, torch.zeros_like(x), -x, -y, x, torch.zeros_like(x)), -1).reshape(log_rot.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=log_rot.device, dtype=log_rot.dtype).expand(log_rot.shape[:-1] + (3, 3))
    return eye + a * k + b * (k @ k)
