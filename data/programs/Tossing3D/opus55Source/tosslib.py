"""Kinematics helpers for Tossing3D (private MuJoCo model copy)."""
import numpy as np
import mujoco

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from kinder.envs.dynamic3d.task_families import Tossing3DEnv
        env = Tossing3DEnv(num_objects=1, scene_bg=False)
        _MODEL = env._object_centric_env._robot_env.sim.model.mj_model
    return _MODEL


class Kin:
    def __init__(self):
        self.m = get_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.base_adr = [m.jnt_qposadr[jid(n)] for n in ("robot_joint_x", "robot_joint_y", "robot_joint_th")]
        self.arm_adr = [m.jnt_qposadr[jid(f"robot_joint_{i}")] for i in range(1, 8)]
        self.arm_dof = [m.jnt_dofadr[jid(f"robot_joint_{i}")] for i in range(1, 8)]
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def fk(self, base, q):
        d = self.d
        for a, v in zip(self.base_adr, base):
            d.qpos[a] = v
        for a, v in zip(self.arm_adr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, base, q0, pos, rot=None, iters=300, tol=1e-4, rot_w=0.3, qnom=None, null_w=0.1):
        q = np.array(q0, dtype=float)
        qnom = q.copy() if qnom is None else np.array(qnom, dtype=float)
        lo = np.array([-1e9, -2.24, -1e9, -2.57, -1e9, -2.09, -1e9])
        hi = -lo
        for _ in range(iters):
            p, R = self.fk(base, q)
            ep = pos - p
            err = [ep]
            if rot is not None:
                er = 0.5 * (np.cross(R[:, 0], rot[:, 0]) + np.cross(R[:, 1], rot[:, 1]) + np.cross(R[:, 2], rot[:, 2]))
                err.append(rot_w * er)
            e = np.concatenate(err)
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = [self.jacp[:, self.arm_dof]]
            if rot is not None:
                J.append(rot_w * self.jacr[:, self.arm_dof])
            J = np.vstack(J)
            lam = 1e-3
            Jp = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(J.shape[0]))
            dq = Jp @ e
            dq += null_w * (np.eye(7) - Jp @ J) @ (qnom - q)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, lo, hi)
        p, R = self.fk(base, q)
        return q, np.linalg.norm(pos - p)
