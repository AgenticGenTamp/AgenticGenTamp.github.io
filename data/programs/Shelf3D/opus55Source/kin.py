"""Kinematics helpers for the TidyBot (Kinova Gen3) using the MuJoCo model."""
import os
import numpy as np
import mujoco

_XML = "/robocode/third-party/kindergarden/src/kinder/envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
if not os.path.exists(_XML):
    import kinder.envs.dynamic3d as _kd
    _XML = os.path.join(os.path.dirname(_kd.__file__), "models", "stanford_tidybot", "tidybot.xml")

CUBE_OFF = 0.015  # cube center is this far along gripper approach axis from pinch site


class Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(_XML)
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        m = self.m
        self.arm_geoms = [g for g in range(m.ngeom)
                          if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] > 0
                          and m.body(m.geom_bodyid[g]).name != "base_link"]
        self._verts = {}
        for g in self.arm_geoms:
            if m.geom_type[g] == 7:
                mid = m.geom_dataid[g]
                va = m.mesh_vertadr[mid]; vn = m.mesh_vertnum[mid]
                self._verts[g] = m.mesh_vert[va:va + vn].copy()
            else:
                s = m.geom_size[g]
                self._verts[g] = np.array([[a, b, c] for a in (-s[0], s[0]) for b in (-s[1], s[1]) for c in (-s[2], s[2])])
        self.jac_p = np.zeros((3, m.nv)); self.jac_r = np.zeros((3, m.nv))

    def fk(self, q):
        d = self.d
        d.qpos[:] = 0; d.qpos[3:10] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def cube_center(self, q):
        p, R = self.fk(q)
        return p + CUBE_OFF * R[:, 2], R

    def arm_points(self, q):
        """All collision-vertex points of the arm (base frame), list per geom."""
        self.fk(q)
        out = []
        for g in self.arm_geoms:
            R = self.d.geom_xmat[g].reshape(3, 3)
            out.append(self._verts[g] @ R.T + self.d.geom_xpos[g])
        return out

    def ik(self, pos_c, Rt, seed, iters=200, tol=1e-4, null_target=None):
        """Solve for q such that cube center = pos_c and site rotation = Rt (base frame)."""
        q = np.array(seed, dtype=float)
        m, d = self.m, self.d
        for it in range(iters):
            p, R = self.cube_center(q)
            ep = pos_c - p
            er = 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1]) + np.cross(R[:, 2], Rt[:, 2]))
            err = np.concatenate([ep, er])
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < tol * 5:
                return q, True
            # jacobian of point cube center: use site jac + offset
            mujoco.mj_jacSite(m, d, self.jac_p, self.jac_r, self.sid)
            Jp = self.jac_p[:, 3:10]; Jr = self.jac_r[:, 3:10]
            off = CUBE_OFF * R[:, 2]
            Jp = self.jac_p[:, 3:10] + np.cross(Jr.T, off[None, :]).T
            J = np.vstack([Jp, Jr])
            lam = 1e-3
            JJt = J @ J.T + lam * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            if null_target is not None:
                N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
                dq += N @ (0.1 * (null_target - q))
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q = q + dq
        p, R = self.cube_center(q)
        ok = np.linalg.norm(pos_c - p) < 1e-3
        return q, ok


def rot_pitch(th, sign=1.0):
    """Gripper rotation: approach z=(sin th,0,-cos th), finger axis x=(0,sign,0)."""
    z = np.array([np.sin(th), 0.0, -np.cos(th)])
    x = np.array([0.0, sign, 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
