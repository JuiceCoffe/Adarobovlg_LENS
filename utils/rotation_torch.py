import torch
from scipy.spatial.transform import Rotation


def matrix_to_euler(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix/matrices to intrinsic XYZ Euler angles.

    This function supports both single and batched inputs. The shape
    is preserved in a natural way via SciPy's broadcasting:

        (3, 3)       -> (3,)
        (B, 3, 3)    -> (B, 3)
        (..., 3, 3)  -> (..., 3)

    Args:
        matrix (torch.Tensor): Rotation matrix of shape (..., 3, 3) or (3, 3).

    Returns:
        torch.Tensor: Euler angles in radians of shape (..., 3) or (3,).
    """
    device = matrix.device
    euler = Rotation.from_matrix(matrix.cpu().numpy()).as_euler("XYZ")
    return torch.tensor(euler, dtype=torch.float32, device=device)


def euler_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert intrinsic XYZ Euler angles to rotation matrix.

    This function supports both single and batched inputs, relying on
    SciPy's broadcasting:

        (3,)       -> (3, 3)
        (B, 3)     -> (B, 3, 3)
        (..., 3)   -> (..., 3, 3)

    Args:
        euler (torch.Tensor): Euler angles in radians of shape (..., 3) or (3,).

    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3) or (3, 3).
    """
    device = euler.device
    matrix = Rotation.from_euler("XYZ", euler.cpu().numpy()).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix/matrices to 6D rotation representation.

    The 6D representation is constructed from the first two columns
    of the rotation matrix. Implementation-wise we transpose and then
    flatten the last two dimensions and take the first 6 elements.

    Shapes:
        (3, 3)        -> (6,)
        (B, 3, 3)     -> (B, 6)
        (..., 3, 3)   -> (..., 6)

    Args:
        matrix (torch.Tensor): Rotation matrix of shape (..., 3, 3) or (3, 3).

    Returns:
        torch.Tensor: 6D rotation representation of shape (..., 6) or (6,).
    """
    # transpose last two dims, then flatten them
    mat_T = matrix.mT  # (..., 3, 3)
    flat = mat_T.reshape(*mat_T.shape[:-2], 9)  # (..., 9)
    return flat[..., :6]


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix/matrices.

    Shapes:
        (6,)       -> (3, 3)
        (B, 6)     -> (B, 3, 3)
        (..., 6)   -> (..., 3, 3)

    Args:
        rot6d (torch.Tensor): 6D rotation representation of shape (..., 6) or (6,).

    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3) or (3, 3).
    """
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(torch.cross(x, y, dim=-1))

    matrix = torch.stack([x, y, z], dim=-2).mT  # (..., 3, 3)
    return matrix


def euler_to_rot6d(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to 6D rotation representation.

    Shapes:
        (3,)       -> (6,)
        (B, 3)     -> (B, 6)
        (..., 3)   -> (..., 6)

    Args:
        euler (torch.Tensor): Euler angles in radians of shape (..., 3) or (3,).

    Returns:
        torch.Tensor: 6D rotation representation of shape (..., 6) or (6,).
    """
    matrix = euler_to_matrix(euler)
    return matrix_to_rot6d(matrix)


def rot6d_to_euler(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to Euler angles.

    Shapes:
        (6,)       -> (3,)
        (B, 6)     -> (B, 3)
        (..., 6)   -> (..., 3)

    Args:
        rot6d (torch.Tensor): 6D rotation representation of shape (..., 6) or (6,).

    Returns:
        torch.Tensor: Euler angles in radians of shape (..., 3) or (3,).
    """
    matrix = rot6d_to_matrix(rot6d)
    return matrix_to_euler(matrix)


def axisangle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix/matrices.

    Both axis and angle support broadcasting. The axis is normalized
    internally.

    Typical shapes:
        axis:  (3,),   angle: ()         -> (3, 3)
        axis:  (B, 3), angle: (B,)       -> (B, 3, 3)
        axis:  (..., 3), angle: (...)    -> (..., 3, 3)

    Args:
        axis (torch.Tensor): Rotation axis of shape (..., 3) or (3,).
        angle (torch.Tensor): Rotation angle in radians of shape (...) or scalar.

    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3) or (3, 3).
    """
    axis = axis.to(dtype=torch.float32)
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)

    angle = torch.as_tensor(angle, dtype=axis.dtype, device=axis.device)

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = torch.cos(angle), torch.sin(angle)

    m00 = (1 - c) * x * x + c
    m01 = (1 - c) * x * y - s * z
    m02 = (1 - c) * x * z + s * y

    m10 = (1 - c) * x * y + s * z
    m11 = (1 - c) * y * y + c
    m12 = (1 - c) * y * z - s * x

    m20 = (1 - c) * x * z - s * y
    m21 = (1 - c) * y * z + s * x
    m22 = (1 - c) * z * z + c

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    matrix = torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)

    return matrix


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert intrinsic XYZ Euler angles to quaternion(s).

    Shapes:
        (3,)       -> (4,)
        (B, 3)     -> (B, 4)
        (..., 3)   -> (..., 4)

    Args:
        euler (torch.Tensor): Euler angles in radians of shape (..., 3) or (3,).

    Returns:
        torch.Tensor: Quaternions of shape (..., 4) or (4,) in (x, y, z, w) order.
    """
    device = euler.device
    quat = Rotation.from_euler("XYZ", euler.cpu().numpy()).as_quat()
    return torch.tensor(quat, dtype=torch.float32, device=device)


def normalize(v: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor along the last dimension using L2 norm.

    Args:
        v (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor with the same shape as input.
    """
    return v / torch.norm(v, dim=-1, keepdim=True)


def q_euler_to_q_rot6d(q_euler: torch.Tensor) -> torch.Tensor:
    """
    Replace the Euler-angle segment in a configuration vector with a 6D rotation
    representation.

    The last dimension is assumed to be at least 6:
        [..., :3]   - kept as is
        [..., 3:6]  - interpreted as Euler angles and converted to 6D
        [..., 6:]   - kept as is

    Shapes:
        (..., N)    -> (..., N + 3)

    Args:
        q_euler (torch.Tensor): Input tensor containing Euler angles in the
                                last dimension, shape (..., N).

    Returns:
        torch.Tensor: Tensor with Euler angles replaced by 6D representation,
                      shape (..., N + 3).
    """
    head = q_euler[..., :3]
    euler = q_euler[..., 3:6]
    tail = q_euler[..., 6:]
    rot6d = euler_to_rot6d(euler)
    return torch.cat([head, rot6d, tail], dim=-1)


def q_rot6d_to_q_euler(q_rot6d: torch.Tensor) -> torch.Tensor:
    """
    Replace the 6D rotation segment in a configuration vector with Euler angles.

    The last dimension is assumed to be at least 9:
        [..., :3]   - kept as is
        [..., 3:9]  - interpreted as 6D rotation and converted to Euler
        [..., 9:]   - kept as is

    Shapes:
        (..., N)    -> (..., N - 3)

    Args:
        q_rot6d (torch.Tensor): Input tensor containing 6D rotation in the
                                last dimension, shape (..., N).

    Returns:
        torch.Tensor: Tensor with 6D rotation replaced by Euler angles,
                      shape (..., N - 3).
    """
    head = q_rot6d[..., :3]
    rot6d = q_rot6d[..., 3:9]
    tail = q_rot6d[..., 9:]
    euler = rot6d_to_euler(rot6d)
    return torch.cat([head, euler, tail], dim=-1)


if __name__ == "__main__":
    """Quick sanity tests for single and batched inputs."""

    # ----- single (no batch) -----
    e = torch.rand(3) * 2 * torch.pi
    R = euler_to_matrix(e)
    e2 = matrix_to_euler(R)
    R2 = euler_to_matrix(e2)
    assert R.shape == (3, 3)
    assert e2.shape == (3,)
    assert torch.allclose(R, R2, atol=1e-5)

    r6 = matrix_to_rot6d(R)
    R3 = rot6d_to_matrix(r6)
    assert r6.shape == (6,)
    assert R3.shape == (3, 3)
    assert torch.allclose(R, R3, atol=1e-5)

    # ----- batched -----
    e_batch = torch.rand(5, 3) * 2 * torch.pi
    R_batch = euler_to_matrix(e_batch)
    e_batch2 = matrix_to_euler(R_batch)
    assert R_batch.shape == (5, 3, 3)
    assert e_batch2.shape == (5, 3)
    assert torch.allclose(R_batch, euler_to_matrix(e_batch2), atol=1e-5)

    print("All quick tests passed.")
