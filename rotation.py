"""Torque-free rotation and quaternion face normals from the A1 model."""

import numpy as np


def propagate_torque_free_batch(
    q0: np.ndarray,
    omega0_body: np.ndarray,
    i_p: np.ndarray,
    t_grid_s: np.ndarray,
    dt_max_s: np.ndarray,
    t0_s: float | None = None,
):
    """N bodies advanced together on ONE shared time grid. Speed only; BIT-IDENTICAL.

    `t0_s` (banded K, step 1 repair): the epoch at which `q0` / `omega0_body` hold. `None` is the
    legacy reading -- the initial state holds at `t_grid_s[0]` -- and every existing caller that
    passes nothing keeps its bytes. When an epoch is given and differs from `t_grid_s[0]`, the
    bodies are first integrated from the epoch to `t_grid_s[0]` (backward when the first sample
    precedes the epoch), with the same RK4, the same per-body step bound and the same spelling,
    and the arrived state is what `t_grid_s[0]` reports. When the epoch IS `t_grid_s[0]` no
    pre-segment executes at all, which is what keeps the default-count render bit-identical.

    WHY THIS EXISTS. The per-object integrator is a Python `while` loop of RK4 substeps,
    and a fast spinner takes on the order of ten thousand of them across one exposure
    window. At the post-C3 debris population that loop runs once per object with no shared
    work, and the pre-launch render review projects the scene wall well past the manifest
    ceiling on this term alone. RK4's arithmetic for one body depends on no other body, so
    the loop can be run once over arrays of shape (n, 4) and (n, 3) instead of n times over
    shape (4,) and (3,).

    WHY IT IS BIT-IDENTICAL, AND WHAT THAT COST. Every expression below is written in the
    SAME form and the SAME association as the scalar path, because floating-point addition
    and multiplication are not associative and a tidier spelling is a different number. The
    first draft of this function precomputed `(I1 - I2) / I0` once per body instead of
    evaluating `(I1 - I2) * wy * wz / I0` per step: algebraically identical, and it broke
    parity at the 1e-15 level on the first probe. That is the whole reason the parity test
    is `np.array_equal` over a seeded population and not `allclose`.

    PER-BODY STEP SCHEDULES ARE PRESERVED. `dt_max_s` is per body (it derives from the spin
    period), so bodies reach each sample time in different numbers of substeps. The loop
    keeps a per-body clock and an ACTIVE mask, and writes back only where active, so a body
    that has already arrived is not stepped again -- which is what the scalar `while`
    condition does, expressed over an array.

    NOT vectorized across TIME, and it cannot be: RK4's state at step k depends on step
    k-1. The batching axis is the population, which is the axis that grew."""
    q0 = np.asarray(q0, dtype=float)
    omega0_body = np.asarray(omega0_body, dtype=float)
    i_p = np.asarray(i_p, dtype=float)
    dt_max = np.asarray(dt_max_s, dtype=float)
    n = q0.shape[0]
    m = len(t_grid_s)
    q = np.array(q0, dtype=float)
    w = np.array(omega0_body, dtype=float)
    out_q = np.empty((n, m, 4))
    out_w = np.empty((n, m, 3))
    t = np.full(n, float(t_grid_s[0]))
    a0, a1, a2 = i_p[:, 0], i_p[:, 1], i_p[:, 2]

    def deriv(qq, ww):
        # quat_mul(qq, [0, wx, wy, wz]) and _omega_dot(i_p, ww), term for term.
        qw, qx, qy, qz = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
        wx, wy, wz = ww[:, 0], ww[:, 1], ww[:, 2]
        dq = 0.5 * np.stack(
            [
                qw * 0.0 - qx * wx - qy * wy - qz * wz,
                qw * wx + qx * 0.0 + qy * wz - qz * wy,
                qw * wy - qx * wz + qy * 0.0 + qz * wx,
                qw * wz + qx * wy - qy * wx + qz * 0.0,
            ],
            axis=1,
        )
        dw = np.stack(
            [
                (a1 - a2) * wy * wz / a0,
                (a2 - a0) * wz * wx / a1,
                (a0 - a1) * wx * wy / a2,
            ],
            axis=1,
        )
        return dq, dw

    if t0_s is not None and float(t0_s) != float(t_grid_s[0]):
        # THE PRE-SEGMENT: from the reference epoch to the first requested sample, signed. The
        # step is the main loop's step with `dt` allowed a sign; per-body clocks and the active
        # mask exactly as below. This block is skipped entirely when the epoch is the first
        # sample, so a default-count grid runs the bytes it always ran.
        target = float(t_grid_s[0])
        clock = np.full(n, float(t0_s))
        direction = 1.0 if target > float(t0_s) else -1.0
        while True:
            active = direction * (target - clock) > 1e-15
            if not active.any():
                break
            dt = direction * np.minimum(dt_max, np.abs(target - clock))
            dtc = dt[:, None]
            k1q, k1w = deriv(q, w)
            k2q, k2w = deriv(q + 0.5 * dtc * k1q, w + 0.5 * dtc * k1w)
            k3q, k3w = deriv(q + 0.5 * dtc * k2q, w + 0.5 * dtc * k2w)
            k4q, k4w = deriv(q + dtc * k3q, w + dtc * k3w)
            qn = q + dtc / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
            wn = w + dtc / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            qn = qn / np.linalg.norm(qn, axis=1)[:, None]
            mask = active[:, None]
            q = np.where(mask, qn, q)
            w = np.where(mask, wn, w)
            clock = np.where(active, clock + dt, clock)
    out_q[:, 0] = q
    out_w[:, 0] = w

    for k in range(1, m):
        t_target = float(t_grid_s[k])
        while True:
            active = t < t_target - 1e-15
            if not active.any():
                break
            dt = np.minimum(dt_max, t_target - t)
            dtc = dt[:, None]
            k1q, k1w = deriv(q, w)
            k2q, k2w = deriv(q + 0.5 * dtc * k1q, w + 0.5 * dtc * k1w)
            k3q, k3w = deriv(q + 0.5 * dtc * k2q, w + 0.5 * dtc * k2w)
            k4q, k4w = deriv(q + dtc * k3q, w + dtc * k3w)
            qn = q + dtc / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
            wn = w + dtc / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            qn = qn / np.linalg.norm(qn, axis=1)[:, None]
            mask = active[:, None]
            q = np.where(mask, qn, q)
            w = np.where(mask, wn, w)
            t = np.where(active, t + dt, t)
        out_q[:, k] = q
        out_w[:, k] = w
    return out_q, out_w


def _cross3_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross for stacks of 3-vectors, in its own term order."""
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _quat_rotate_batch(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """`quat_rotate` for a stack of quaternions against one fixed body-frame vector.

    Term for term: r = q[1:]; v + 2.0 * cross(r, cross(r, v) + w * v)."""
    w = q[..., 0]
    r = q[..., 1:]
    inner = _cross3_batch(r, np.broadcast_to(v, r.shape)) + w[..., None] * v
    return v + 2.0 * _cross3_batch(r, inner)


def _body_axes_batch(qs: np.ndarray, rot=None):
    """`_body_axes` over a stack of attitudes. See the matvec note in the header above."""
    axes = (
        _quat_rotate_batch(qs, np.array([1.0, 0.0, 0.0])),
        _quat_rotate_batch(qs, np.array([0.0, 1.0, 0.0])),
        _quat_rotate_batch(qs, np.array([0.0, 0.0, 1.0])),
    )
    if rot is None:
        return axes
    r = np.asarray(rot, dtype=float)
    return tuple(np.matmul(r, axis[..., None])[..., 0] for axis in axes)
