import torch
import numpy as np
import torch.nn.functional as F

def to_torch(x, device='cpu'):
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.squeeze(0).cpu().numpy()

@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero sub-gradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )

@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))

@torch.jit.script
def transform_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    rot = matrix_from_quat(quat)

    t = torch.eye(4, device=pos.device).repeat(pos.shape[0], 1, 1)
    t[:, :3, :3] = rot
    t[:, :3, 3] = pos

    return t

@torch.jit.script
def transform_from_spatial_vector(spatial_vector: torch.Tensor) -> torch.Tensor:
    pos = spatial_vector[:, :3]
    rotvec = spatial_vector[:, 3:]

    angle = torch.norm(rotvec, dim=1)

    # Avoid division by zero
    angle = angle + 1e-6

    axis = rotvec / angle.unsqueeze(1)
    quat = quat_from_angle_axis(angle, axis)
    rot = matrix_from_quat(quat)

    return transform_from_pos_quat(pos, quat)

@torch.jit.script
def spatial_vector_from_transform(transform: torch.Tensor) -> torch.Tensor:
    rot = transform[:, :3, :3]
    pos = transform[:, :3, 3]

    quat = quat_from_matrix(rot)
    rotvec = axis_angle_from_quat(quat)

    return torch.cat([pos, rotvec], dim=1)

@torch.jit.script
def apply_dead_band_smooth(component: torch.Tensor, bandwidth: float, smooth_band: float) -> torch.Tensor:
    """
    component: batch of vector of size (N, M)
    bandwidth: float
    smooth_band: float
    """
    abs_component = torch.abs(component)
    mask_dead = abs_component <= bandwidth
    mask_smooth = (abs_component > bandwidth) & (abs_component <= (bandwidth + smooth_band))
    mask_pass = abs_component > (bandwidth + smooth_band)

    smooth_result = torch.zeros_like(component, device=component.device)

    smooth_result[mask_dead] = 0

    smooth_result[mask_pass] = (abs_component[mask_pass] - bandwidth - smooth_band * 0.5) / abs_component[mask_pass]

    s = abs_component[mask_smooth] - bandwidth
    smooth_result[mask_smooth] = (0.5 * s * s / smooth_band) / abs_component[mask_smooth]

    return smooth_result * component

@torch.jit.script
def wrench_dead_band_smooth(wrench_in: torch.Tensor, bandwidth_force: float, bandwidth_torque: float, smooth_force: float, smooth_torque: float) -> torch.Tensor:
    """
    Apply the smooth dead band to each component of the wrench input
    (Torch version)
    wrench_in: 6D vector of size (N, 6)
    bandwidth_force: float
    bandwidth_torque: float
    smooth_force: float
    smooth_torque: float
    """
    assert wrench_in.shape[1] == 6, "wrench_in must be a 6D vector"

    # Separate force and torque components
    force = wrench_in[:, :3]
    torque = wrench_in[:, 3:]

    # Apply smoothing to force and torque components
    smooth_force_result = apply_dead_band_smooth(force, bandwidth_force, smooth_force)
    smooth_torque_result = apply_dead_band_smooth(torque, bandwidth_torque, smooth_torque)

    # Combine the results
    wrench_out = torch.cat((smooth_force_result, smooth_torque_result), dim=1)

    return wrench_out

@torch.jit.script
def wrench_trans(T_from_to: torch.Tensor, w_from: torch.Tensor) -> torch.Tensor:
    """
    Transforms a wrench to a new point of view.

    Args:
    T_from_to: The transformation to the new point of view (Pose) 
                represented as a 4x4 homogeneous transformation matrix. (N, 4, 4)
    w_from: Wrench to transform in list format (N, 6) [F_x, F_y, F_z, M_x, M_y, M_z]

    Returns:
    resulting wrench, w_to in list format (N, 6) [F_x, F_y, F_z, M_x, M_y, M_z]
    """

    T_inv = torch.inverse(T_from_to)
    t0 = T_inv[:, :3, 3]
    r0 = T_inv[:, :3, :3]

    F = w_from[:, :3]
    M = w_from[:, 3:]

    F_to = torch.matmul(r0, F.unsqueeze(2)).squeeze(2)
    M_to = torch.matmul(r0, M.unsqueeze(2)).squeeze(2) + torch.cross(t0, F_to, dim=1)
    w_to = torch.cat((F_to, M_to), dim=1)
    return w_to

@torch.jit.script
def adm_rotate_velocity_in_frame(frame: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """
    Rotates a velocity into a new reference frame.

    :param frame: List or array of the current velocity reference frame (N, 4, 4)
    :param velocity: List or array of the input velocity vector (N, 6) [vx, vy, vz, vrx, vry, vrz]
    :returns: List of the velocity in the new reference frame (N, 6) [vx, vy, vz, vrx, vry, vrz]
    """
    r0 = frame[:, :3, :3]
    vel_pos = velocity[:, :3]
    vel_rot = velocity[:, 3:]

    vel_pos_new = torch.matmul(r0, vel_pos.unsqueeze(2)).squeeze(2)
    vel_rot_new = torch.matmul(r0, vel_rot.unsqueeze(2)).squeeze(2)

    return torch.cat((vel_pos_new, vel_rot_new), dim=1)

@torch.jit.script
def adm_rotate_wrench_in_frame(frame: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    return adm_rotate_velocity_in_frame(frame, wrench)

@torch.jit.script
def pose_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """
    Calculate the pose error between two poses.
    desired: The desired pose (4x4 matrix) (N, 4, 4)
    current: The current pose (4x4 matrix) (N, 4, 4)
    """
    rc1 = current[:, 0:3, 0]
    rc2 = current[:, 0:3, 1]
    rc3 = current[:, 0:3, 2]
    rd1 = desired[:, 0:3, 0]
    rd2 = desired[:, 0:3, 1]
    rd3 = desired[:, 0:3, 2]

    error = torch.zeros((desired.shape[0], 6), device=desired.device)
    error[:, 0:3] = desired[:, 0:3, 3] - current[:, 0:3, 3]
    error[:, 3:6] = 0.5 * (torch.cross(rc1, rd1, dim=1) + torch.cross(rc2, rd2, dim=1) + torch.cross(rc3, rd3, dim=1))

    return error

@torch.jit.script
def adm_vel_trans(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_swap = torch.zeros_like(v, device=v.device)
    v_swap[:, :3] = v[:, 3:]
    v_swap[:, 3:] = v[:, :3]

    vw = wrench_trans(t, v_swap)

    vw_swap = torch.zeros_like(vw, device=vw.device)
    vw_swap[:, :3] = vw[:, 3:]
    vw_swap[:, 3:] = vw[:, :3]

    return vw_swap

@torch.jit.script
def apply_delta_transform(transform: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    delta_transform = transform_from_spatial_vector(delta)
    return torch.matmul(transform, delta_transform)

@torch.jit.script
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

@torch.jit.script
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

@torch.jit.script
def rotation_6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(matrix_from_quat(quat))

@torch.jit.script
def quat_from_rotation_6d(d6: torch.Tensor) -> torch.Tensor:
    return quat_from_matrix(rotation_6d_to_matrix(d6))

def pose_9d_from_transform(matrix : torch.Tensor) -> torch.Tensor:
    translation = matrix[:, :3, 3]
    rotation = matrix_to_rotation_6d(matrix[:, :3, :3])

    return torch.cat([translation, rotation], dim=1)

def posquat_from_transform(matrix: torch.Tensor) -> torch.Tensor:
    translation = matrix[:, :3, 3]
    rotation = quat_from_matrix(matrix[:, :3, :3])

    return torch.cat([translation, rotation], dim=1)

def transform_from_posquat(posquat: torch.Tensor) -> torch.Tensor:
    translation = posquat[:, :3]
    rotation = matrix_from_quat(posquat[:, 3:])

    matrix = torch.eye(4).repeat(posquat.shape[0], 1, 1)
    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = translation

    return matrix

def transform_from_pose_9d(pose: torch.Tensor) -> torch.Tensor:
    translation = pose[:, :3]
    rotation = rotation_6d_to_matrix(pose[:, 3:])

    matrix = torch.eye(4).repeat(pose.shape[0], 1, 1)
    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = translation

    return matrix