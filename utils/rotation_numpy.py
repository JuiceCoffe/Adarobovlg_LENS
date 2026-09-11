import numpy as np
from scipy.spatial.transform import Rotation


def inv_virtual_joints(T):
    # T is a 4x4 homogeneous transform from world to base_link.
    """Extract intrinsic XYZ Euler angles (roll, pitch, yaw), in radians, from the transform rotation."""
    R = T[:3, :3]

    # Orientation.
    pitch = np.arcsin(R[0, 2])
    roll = np.arctan2(-R[1, 2], R[2, 2])
    yaw = np.arctan2(-R[0, 1], R[0, 0])

    return roll, pitch, yaw


def matrix_to_euler(matrix, rule="XYZ"):
    """
    Convert rotation matrix to XYZ intrinsic Euler angles.

    Args:
        matrix: array-like (..., 3, 3)

    Returns:
        Euler angles (..., 3) in radians.
    """
    matrix = np.asarray(matrix)
    euler = Rotation.from_matrix(matrix).as_euler(rule)
    return euler.astype(np.float32)


def euler_to_matrix(euler, rule="XYZ"):
    """
    Convert XYZ intrinsic Euler angles to rotation matrix.

    Args:
        euler: array-like (..., 3)

    Returns:
        Rotation matrix (..., 3, 3).
    """
    euler = np.asarray(euler)
    matrix = Rotation.from_euler(rule, euler).as_matrix()
    return matrix.astype(np.float32)


def matrix_to_rot6d(matrix):
    """
    Convert a 3x3 rotation matrix to 6D rotation representation.

    Args:
        matrix: array-like (3, 3)

    Returns:
        (6,) rot6d vector.
    """
    matrix = np.asarray(matrix)
    return matrix.T.reshape(-1)[:6].astype(np.float32)


def normalize(v):
    """
    L2 normalize along the last dimension.

    Args:
        v: array-like (..., n)

    Returns:
        Normalized vector (..., n).
    """
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm


def rot6d_to_matrix(rot6d):
    """
    Convert 6D rotation representation back to a rotation matrix.

    Args:
        rot6d: array-like (..., 6)

    Returns:
        Rotation matrix (..., 3, 3).
    """
    rot6d = np.asarray(rot6d, dtype=np.float32)

    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(np.cross(x, y, axis=-1))

    matrix = np.stack([x, y, z], axis=-2)
    matrix = np.swapaxes(matrix, -1, -2)  # transpose last two dims
    return matrix


def euler_to_rot6d(euler):
    """
    Convert Euler angles to 6D rotation representation.

    Args:
        euler: array-like (..., 3)

    Returns:
        (..., 6) rot6d representation.
    """
    matrix = euler_to_matrix(euler)
    return matrix_to_rot6d(matrix)


def rot6d_to_euler(rot6d):
    """
    Convert 6D rotation representation to Euler angles.

    Args:
        rot6d: array-like (..., 6)

    Returns:
        (..., 3) Euler angles.
    """
    matrix = rot6d_to_matrix(rot6d)
    return matrix_to_euler(matrix)


def axisangle_to_matrix(axis, angle):
    """
    Convert axis-angle rotation to rotation matrix.

    Args:
        axis: array-like (3,) rotation axis
        angle: float rotation angle in radians

    Returns:
        (3, 3) rotation matrix.
    """
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [(1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
            [(1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x],
            [(1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c],
        ],
        dtype=np.float32,
    )


def euler_to_quaternion(euler):
    """
    Convert Euler angles to quaternion.

    Args:
        euler: array-like (..., 3)

    Returns:
        Quaternion (..., 4) in (x, y, z, w) order.
    """
    euler = np.asarray(euler)
    quaternion = Rotation.from_euler("XYZ", euler).as_quat()
    return quaternion.astype(np.float32)


def q_euler_to_q_rot6d(q_euler):
    """
    Replace the Euler segment of a vector with its 6D rotation representation.

    Args:
        q_euler: array-like (..., N)

    Returns:
        Updated vector (..., N) with 6D rotation replacing Euler angles.
    """
    q_euler = np.asarray(q_euler, dtype=np.float32)
    head = q_euler[..., :3]
    euler = q_euler[..., 3:6]
    tail = q_euler[..., 6:]

    rot6d = euler_to_rot6d(euler)
    return np.concatenate([head, rot6d, tail], axis=-1)


def q_rot6d_to_q_euler(q_rot6d):
    """
    Replace the 6D rotation segment of a vector with Euler angles.

    Args:
        q_rot6d: array-like (..., N)

    Returns:
        Updated vector (..., N) with Euler replacing 6D rotation.
    """
    q_rot6d = np.asarray(q_rot6d, dtype=np.float32)
    head = q_rot6d[..., :3]
    rot6d = q_rot6d[..., 3:9]
    tail = q_rot6d[..., 9:]

    euler = rot6d_to_euler(rot6d)
    return np.concatenate([head, euler, tail], axis=-1)


if __name__ == "__main__":
    """Test correctness of the above functions. Euler equality is not strictly required due to singularities."""
    test_euler = np.random.rand(3).astype(np.float32) * 2 * np.pi

    # Euler <-> Matrix
    test_matrix = euler_to_matrix(test_euler)
    test_euler_prime = matrix_to_euler(test_matrix)
    test_matrix_prime = euler_to_matrix(test_euler_prime)
    assert np.allclose(
        test_matrix, test_matrix_prime
    ), f"Original Matrix: {test_matrix}, Converted Matrix: {test_matrix_prime}"

    # Matrix <-> Rot6D
    test_rot6d = matrix_to_rot6d(test_matrix)
    test_matrix_prime = rot6d_to_matrix(test_rot6d)
    assert np.allclose(
        test_matrix, test_matrix_prime
    ), f"Original Matrix: {test_matrix}, Converted Matrix: {test_matrix_prime}"

    test_rot6d_prime = matrix_to_rot6d(test_matrix_prime)
    assert np.allclose(
        test_rot6d, test_rot6d_prime
    ), f"Original Rot6D: {test_rot6d}, Converted Rot6D: {test_rot6d_prime}"

    # Rot6D <-> Euler
    test_euler_prime = rot6d_to_euler(test_rot6d)
    test_rot6d_prime = euler_to_rot6d(test_euler_prime)
    assert np.allclose(
        test_rot6d, test_rot6d_prime
    ), f"Original Rot6D: {test_rot6d}, Converted Rot6D: {test_rot6d_prime}"

    print("All Tests Passed!")
