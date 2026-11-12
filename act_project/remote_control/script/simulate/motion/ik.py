"""6D DLS 逆运动学（阻尼最小二乘法）。

纯算法，输入：
    forward   —— callable() -> (pos[3], rot[3,3])
    jacobian  —— callable(q) -> (6, nq)  其中前 3 行是位置 J，后 3 行是旋转 J
    q0        —— 初始猜测 [nq]
    target_pos [3]
    target_rot [3,3]
    lam       —— DLS 阻尼（越大越稳但越慢）
    max_iter  —— 最大迭代
    pos_tol   —— 位置收敛阈值（米）
    rot_tol   —— 姿态收敛阈值（弧度，等价旋转向量长度）

返回：
    (q_final, success, iters)
"""

import numpy as np


def dls_ik(
    forward,
    jacobian,
    q0,
    target_pos,
    target_rot,
    lam: float = 0.05,
    max_iter: int = 400,
    pos_tol: float = 2e-3,
    rot_tol: float = 4e-2,
    q_min=None,
    q_max=None,
):
    """通用 6D DLS IK。

    forward() / jacobian() 由调用方提供——仿真里是 mujoco.mj_forward + mj_jac，
    真机里是机械臂自带的正解/雅可比 API。算法本身不关心底层。
    """
    q = np.array(q0, dtype=float)
    ep = np.asarray(target_pos, dtype=float)

    for it in range(max_iter):
        pos, rot = forward(q)
        err_p = ep - pos
        err_r = 0.5 * (
            np.cross(rot[:, 0], target_rot[:, 0])
            + np.cross(rot[:, 1], target_rot[:, 1])
            + np.cross(rot[:, 2], target_rot[:, 2])
        )
        if np.linalg.norm(err_p) < pos_tol and np.linalg.norm(err_r) < rot_tol:
            return q, True, it

        J = jacobian(q)  # (6, nq)
        e = np.concatenate([err_p, err_r])  # (6,)
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
        q = q + dq
        if q_min is not None:
            q = np.maximum(q, q_min)
        if q_max is not None:
            q = np.minimum(q, q_max)

    return q, False, max_iter

