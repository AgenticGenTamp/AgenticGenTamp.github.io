"""Kinematics helper using the env's own MuJoCo model (robot only matters)."""
import numpy as np
import mujoco

ARM_LO = np.array([-np.inf, -2.24, -np.inf, -2.57, -np.inf, -2.09, -np.inf])
ARM_HI = -ARM_LO


class Kin:
    def __init__(self):
        from kinder.envs.dynamic3d.task_families import ScoopPour3DEnv
        env = ScoopPour3DEnv(num_objects=10, scene_bg=False)
        env.reset(seed=0)
        oc = env
        # find mujoco model
        re = None
        for attr in ["_robot_env", "robot_env"]:
            if hasattr(oc, attr):
                re = getattr(oc, attr)
        if re is None and hasattr(oc, "_object_centric_env"):
            re = oc._object_centric_env._robot_env
        self.m = re.sim.model.mj_model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = re.sim.data.mj_data.qpos
        self._env = env
        m = self.m
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.base_q = [m.jnt_qposadr[jid(n)] for n in ["robot_joint_x", "robot_joint_y", "robot_joint_th"]]
        self.arm_q = [m.jnt_qposadr[jid("robot_joint_%d" % i)] for i in range(1, 8)]
        self.arm_v = [m.jnt_dofadr[jid("robot_joint_%d" % i)] for i in range(1, 8)]
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def fk(self, base, arm):
        d = self.d
        d.qpos[self.base_q] = base
        d.qpos[self.arm_q] = arm
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, base, arm0, pos, R=None, iters=100, rot_w=0.3, tol=1e-4):
        q = np.array(arm0, dtype=float)
        for _ in range(iters):
            p, Rc = self.fk(base, q)
            ep = pos - p
            if R is not None:
                er = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
                e = np.concatenate([ep, rot_w * er])
            else:
                e = ep
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            Jp = self.jacp[:, self.arm_v]
            J = np.vstack([Jp, rot_w * self.jacr[:, self.arm_v]]) if R is not None else Jp
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, ARM_LO, ARM_HI)
        return q, np.linalg.norm(e)


def down_R(yaw):
    """Gripper pointing down, finger-closing axis (site x) along world yaw."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0, 0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
