"""Kinematics helper for the TidyBot (Kinova Gen3) using a standalone MuJoCo model."""
import os
import numpy as np
import mujoco

ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
BASE_JOINTS = ["joint_x", "joint_y", "joint_th"]


def _model_path():
    import kinder
    return os.path.join(os.path.dirname(kinder.__file__), "envs", "dynamic3d",
                        "models", "stanford_tidybot", "tidybot.xml")


class Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(_model_path())
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in BASE_JOINTS + ARM_JOINTS])
        self.dadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in ARM_JOINTS])
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.lo = np.array([self.m.jnt_range[self.m.joint(n).id][0] if self.m.jnt_limited[self.m.joint(n).id] else -1e9 for n in ARM_JOINTS])
        self.hi = np.array([self.m.jnt_range[self.m.joint(n).id][1] if self.m.jnt_limited[self.m.joint(n).id] else 1e9 for n in ARM_JOINTS])
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))

    def fk(self, base, arm):
        self.d.qpos[self.qadr[:3]] = base
        self.d.qpos[self.qadr[3:]] = arm
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, base, arm0, pos, R=None, iters=100, tol=1e-4, rot_w=0.3):
        """Damped least squares IK. R: desired site rotation matrix (or None)."""
        q = np.array(arm0, dtype=float).copy()
        for _ in range(iters):
            p, Rc = self.fk(base, q)
            ep = pos - p
            if R is not None:
                # orientation error (axis-angle) R_err = R * Rc^T
                Re = R @ Rc.T
                er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                err = np.concatenate([ep, rot_w * er])
            else:
                err = ep
            if np.linalg.norm(ep) < tol and (R is None or np.linalg.norm(er) < 1e-3):
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            J = self.jacp[:, self.dadr]
            if R is not None:
                J = np.vstack([J, rot_w * self.jacr[:, self.dadr]])
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, Rc = self.fk(base, q)
        return q, np.linalg.norm(pos - p)


def down_rot(yaw):
    """Site rotation with z axis pointing down, and x axis rotated by yaw about world z."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
