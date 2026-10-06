"""Kinematics helper for TidyBot (Kinova Gen3) using MuJoCo model."""
import numpy as np
import mujoco

ARM_QADR = 55
ARM_DADR = 49
BASE_QADR = 52
LIM = np.array([[-1e9, 1e9], [-2.2, 2.2], [-1e9, 1e9], [-2.5, 2.5],
                [-1e9, 1e9], [-2.05, 2.05], [-1e9, 1e9]])
Q_HOME = np.array([0, 0.26, np.pi, -2.27, 0, 0.96, np.pi / 2])
Q_RETRACT = np.array([0, -0.349, 3.1416, -2.548, 0, -0.873, 1.5708])


def rot_err(R, Rt):
    return 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1])
                  + np.cross(R[:, 2], Rt[:, 2]))


class Kin:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                     'robot_pinch_site')
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def fk(self, base, q):
        d = self.d
        d.qpos[BASE_QADR:BASE_QADR + 3] = base
        d.qpos[ARM_QADR:ARM_QADR + 7] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, base, q0, pt, Rt, iters=60, wrot=0.5, tol=2e-4):
        q = np.array(q0, dtype=float)
        ep = er = None
        for i in range(iters):
            p, R = self.fk(base, q)
            ep = pt - p
            er = rot_err(R, Rt) if Rt is not None else np.zeros(3)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            if Rt is not None:
                J = np.vstack([self.jp[:, ARM_DADR:ARM_DADR + 7],
                               wrot * self.jr[:, ARM_DADR:ARM_DADR + 7]])
                e = np.concatenate([ep, wrot * er])
                lam = 1e-3
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            else:
                J = self.jp[:, ARM_DADR:ARM_DADR + 7]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), ep)
            # mild pull toward joint centre for limited joints
            dq = np.clip(dq, -0.3, 0.3)
            q = np.clip(q + dq, LIM[:, 0], LIM[:, 1])
        ok = np.linalg.norm(ep) < 3e-3 and np.linalg.norm(er) < 2e-2
        return q, ok, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def ik_multi(self, base, q_seed, pt, Rt, extra_seeds=()):
        best = None
        seeds = [q_seed] + list(extra_seeds) + [Q_HOME, Q_RETRACT]
        for s in seeds:
            q, ok, e1, e2 = self.ik(base, s, pt, Rt)
            if ok:
                # wrap continuous joints to be near the seed
                for j in (0, 2, 4, 6):
                    q[j] = q_seed[j] + ((q[j] - q_seed[j] + np.pi) % (2 * np.pi) - np.pi)
                cost = np.sum((q - q_seed) ** 2)
                if best is None or cost < best[1]:
                    best = (q, cost)
                if s is q_seed:
                    break
        return None if best is None else best[0]


def make_model():
    from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
    env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
    env.reset(seed=0)
    re = env._kinder_env._object_centric_env._robot_env
    return env, re.sim.model.mj_model
