import torch


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack((z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z), -1).reshape(v.shape[:-1] + (3, 3))


class Transform3d:
    def __init__(self, device=None, dtype=torch.float32, matrix=None):
        self._matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(0) if matrix is None else matrix

    def rotate(self, matrix):
        matrix = matrix.to(device=self._matrix.device, dtype=self._matrix.dtype)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        m = torch.eye(4, device=matrix.device, dtype=matrix.dtype).expand(matrix.shape[0], -1, -1).clone()
        m[:, :3, :3] = matrix
        self._matrix = m @ self._matrix
        return self

    def translate(self, translation):
        translation = translation.to(device=self._matrix.device, dtype=self._matrix.dtype)
        if translation.ndim == 1:
            translation = translation.unsqueeze(0)
        m = torch.eye(4, device=translation.device, dtype=translation.dtype).expand(translation.shape[0], -1, -1).clone()
        m[:, :3, 3] = translation
        self._matrix = m @ self._matrix
        return self

    def transform_points(self, points):
        p = points.to(device=self._matrix.device, dtype=self._matrix.dtype)
        return torch.matmul(p, self._matrix[..., :3, :3].transpose(-1, -2)) + self._matrix[..., :3, 3]

    def get_matrix(self):
        return self._matrix


from .so3 import so3_exp_map

__all__ = ["Transform3d", "so3_exp_map"]
