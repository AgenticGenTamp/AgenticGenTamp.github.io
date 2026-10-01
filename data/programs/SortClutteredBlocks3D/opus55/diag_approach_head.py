"""Sort cluttered cubes into color bins with a tidybot (holonomic base + Kinova Gen3).

Strategy: keep the arm in a fixed "gripper pointing down" configuration at a fixed
reach in front of the base; the (fast) base does all horizontal positioning and yaw
alignment, while the (slow) arm only moves up and down (plus joint 7 for gripper yaw).
"""
import numpy as np

from kin import fk_arm, ik

MOUNT = np.array([0.1199, 0.0, 0.3947])
DOWN_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
HOME = np.array([0.0, -0.349, 3.142, -2.548, 0.0, -0.873, 1.571])
COLORS = ["red", "green", "blue", "yellow"]

REACH = 0.56            # arm-frame x of the bracelet
GRASP_DZ = 0.2134       # bracelet world z above cube center for grasp
HOLD_TOOL = 0.219       # bracelet -> held cube center
CARRY_Z = 0.75          # bracelet world z while carrying
CLEAR_Z = 0.515         # held cube z needed before moving over bin walls
BASE_XMIN = 0.56
K_ARM = 1.3
DMAX = 0.2              # max base yaw deviation from pi
G_OPEN = 0.45           # pre-grasp gripper opening command
OPEN_GAP = 0.085 * (1 - G_OPEN)
FINGER_T = 0.012         # finger thickness along closing axis
FINGER_W = 0.0115        # finger pad half width


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_half(a):
    """wrap to [-pi/2, pi/2)"""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def rotz2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def quat_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cube_face_yaw(quat):
    """Yaw (mod pi/2) of the cube's horizontal face normals, and tilt (0=flat)."""
    R = quat_R(*quat)
    up = int(np.argmax(np.abs(R[2, :])))
    tilt = np.arccos(min(1.0, abs(R[2, up])))
    others = [i for i in range(3) if i != up]
    a = R[:, others[0]]
    yaw = np.arctan2(a[1], a[0])
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4, tilt


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self._qcache = {}
        self._qprev = HOME.copy()
        self.reset(None, None)

    # ------------------------------------------------------------------ utils
    def _arm_q(self, zworld, gamma=0.0):
        key = round(float(zworld), 3)
        if key not in self._qcache:
            seed = self._qprev
            if self._qcache:
                k = min(self._qcache, key=lambda kk: abs(kk - key))
                seed = self._qcache[k]
            q, _ = ik(seed, np.array([REACH, 0.0, key - MOUNT[2]]), DOWN_R, tool=0.0, iters=300)
            self._qcache[key] = q
        q = self._qcache[key].copy()
        q[6] += gamma
        return q

    def _parse(self, state):
        robot = None
        cubes, bins = {}, {}
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            if name == "robot":
                robot = o
            elif name.startswith("cube"):
                cubes[name] = o
            elif name.startswith("bin_"):
                bins[name[4:]] = o
        g = lambda o, f: float(state.get(o, f))
        self.q = np.array([g(robot, "pos_arm_joint%d" % i) for i in range(1, 8)])
        self.base = np.array([g(robot, "pos_base_x"), g(robot, "pos_base_y"), g(robot, "pos_base_rot")])
        self.basevel = np.array([g(robot, "vel_base_x"), g(robot, "vel_base_y"), g(robot, "vel_base_rot")])
        self.cubes = {}
        for n, o in cubes.items():
            self.cubes[n] = dict(
                p=np.array([g(o, "x"), g(o, "y"), g(o, "z")]),
                quat=(g(o, "qw"), g(o, "qx"), g(o, "qy"), g(o, "qz")),
            )
        self.bins = {c: np.array([g(o, "x"), g(o, "y"), g(o, "z")]) for c, o in bins.items()}

    def _cube_color(self, name):
        n = len(self.cube_names)
        i = self.cube_rank[name]
        return COLORS[min(3, (i * 4) // max(1, n))]

    def _bracelet_world(self):
        tip = fk_arm(self.q, 0.0)[0]
        p = MOUNT + tip
        xy = self.base[:2] + rotz2(self.base[2]) @ p[:2]
        return np.array([xy[0], xy[1], p[2]])

    def _base_target(self, xy, rot):
        off = rotz2(rot) @ np.array([MOUNT[0] + REACH, 0.0])
        return np.array([xy[0] - off[0], xy[1] - off[1], rot])

    def _which_bin(self, p):
        for c, b in self.bins.items():
            if abs(p[0] - b[0]) < 0.05 and abs(p[1] - b[1]) < 0.05 and p[2] < b[2] + 0.1:
                return c
        return None

    def _in_bin(self, name):
        return self._which_bin(self.cubes[name]["p"]) == self._cube_color(name)

    def _delta_max(self, txy):
        # base must stay at x >= BASE_XMIN: tx + L cos(d) >= BASE_XMIN
        L = MOUNT[0] + REACH
        c = (BASE_XMIN - txy[0]) / L
        if c <= -1:
            return np.pi
        if c >= 1:
            return 0.0
        return min(DMAX, np.arccos(c))

    # ------------------------------------------------------------------ api
    def reset(self, state, info):
        self.phase = "select"
        self.target = None
        self.counter = 0
        self.grip = 0.0
        self.fail = {}
        self.cube_names = []
        self.phi = 0.0
        self.gam_target = 0.0
        self.qi = np.zeros(7)
        self.bi = np.zeros(3)
        if state is not None:
            self._parse(state)
            self.cube_names = sorted(self.cubes, key=lambda s: int(s[4:]))
            self.cube_rank = {n: i for i, n in enumerate(self.cube_names)}
            self._qprev = self.q.copy()
            self._arm_q(CARRY_Z)

    def _axis_clearance(self, name, phi):
        """Clearance score for closing fingers along world angle phi on cube `name`."""
        c = self.cubes[name]["p"]
        col = self._cube_color(name)
        u = np.array([np.cos(phi), np.sin(phi)])
        v = np.array([-u[1], u[0]])
        worst = 1.0
        for n, o in self.cubes.items():
            if n == name:
                continue
            d = o["p"] - c
            if d[2] < -0.015 or np.linalg.norm(d[:2]) > 0.12:
                continue
            if self._which_bin(o["p"]) is not None:
                continue
            if self._cube_color(n) == col and abs(d[2]) < 0.01:
                continue  # grabbing a same-colour neighbour too is harmless
            yaw_n, _ = cube_face_yaw(o["quat"])
            a = yaw_n - phi
            h_u = 0.01 * (abs(np.cos(a)) + abs(np.sin(a)))
            h_v = h_u  # symmetric for a square
            al, pe = abs(d[:2] @ u), abs(d[:2] @ v)
            ex_al = al - h_u - (OPEN_GAP / 2 + FINGER_T)
            ex_pe = pe - h_v - FINGER_W
            worst = min(worst, max(ex_al, ex_pe))
        return worst

    def _gamma(self):
        return wrap(self.q[6] - self._arm_q(CARRY_Z)[6])

    def _select(self):
        best, bestv = None, None
        cur = self._bracelet_world()
        gam = self._gamma()
        for n in self.cube_names:
            if self._in_bin(n):
                continue
            c = self.cubes[n]
            yaw, tilt = cube_face_yaw(c["quat"])
            # topmost cubes: anything resting on it?
            covered = any(o["p"][2] > c["p"][2] + 0.012 and np.linalg.norm(o["p"][:2] - c["p"][:2]) < 0.022
                          for m, o in self.cubes.items() if m != n)
            dmax = self._delta_max(c["p"][:2])
            for k in range(2):
                phi = yaw + k * np.pi / 2
                clr = self._axis_clearance(n, phi)
                # base yaw delta needed with current gamma
                need = wrap_half(phi - np.pi / 2 - gam - np.pi)
                rot_cost = max(0.0, abs(need) - dmax)
                v = ((3.0 - 20 * clr if clr < 0 else 0.0) - c["p"][2] * 5 + (1.0 if covered else 0.0)
                     + 0.2 * np.linalg.norm(c["p"][:2] - cur[:2]) + 0.4 * rot_cost
                     + 0.5 * self.fail.get(n, 0) + 0.2 * tilt)
                if bestv is None or v < bestv:
                    best, bestv = (n, phi), v
        return best

    def _yaw_plan(self, txy):
        """Fix a gamma target at selection time; base rot = pi + remaining yaw."""
        dmax = self._delta_max(txy)
        gam = self._gamma()
        need = wrap_half(self.phi - np.pi / 2 - gam - np.pi)
        d = float(np.clip(need, -dmax, dmax))
        self.gam_target = gam + (need - d)

    def _base_rot(self):
        need = wrap_half(self.phi - np.pi / 2 - self.gam_target - np.pi)
        return np.pi + need

    def get_action(self, state):
        self._parse(state)
        act = np.zeros(11, dtype=np.float32)
        arm_z = CARRY_Z
        gam_t = self.gam_target
        base_t = None

        if self.phase == "select":
            sel = self._select()
            if sel is None:
                self.phase = "done"
            else:
                self.target, self.phi = sel
                self._yaw_plan(self.cubes[self.target]["p"][:2])
                self.phase = "goto_pick"
                self.counter = 0

        if self.phase == "done":
            base_t = self.base.copy()
            self.grip = 0.0

        elif self.phase in ("goto_pick", "descend", "close"):
            c = self.cubes[self.target]
            rot, gam_t = self._base_rot(), self.gam_target
            base_t = self._base_target(c["p"][:2], rot)
            bw = self._bracelet_world()
            xy_err = np.linalg.norm(bw[:2] - c["p"][:2])
            yaw_ok = abs(wrap(base_t[2] - self.base[2])) < 0.05 and abs(gam_t - self._gamma()) < 0.05
            if self.phase == "goto_pick":
                self.grip = G_OPEN
                if xy_err < 0.02 and yaw_ok:
                    self.phase = "descend"
                    self.counter = 0
                    self.stall = 0
            if self.phase == "descend":
                gz = c["p"][2] + GRASP_DZ
                # stay a bit above until well aligned
                arm_z = gz if xy_err < 0.006 else max(gz, c["p"][2] + 0.28)
                self.counter += 1
                qd = self._arm_q(gz, gam_t)
                if bw[2] < gz + 0.004 and xy_err < 0.006:
                    self.phase = "close"
                    self.counter = 0
                elif self.counter > 80:
                    self.phase = "close"
                    self.counter = 0
            if self.phase == "close":
                if self.counter == 0:
                    self.grasp_z = c["p"][2] + GRASP_DZ
                    self.grasp_cz = c["p"][2]
                    self.carry_rot = self._base_rot()
                arm_z = self.grasp_z
                base_t = None  # hold base still while closing
                self.grip = 1.0
                self.counter += 1
                if self.counter >= 7:
                    self.phase = "lift"
                    self.counter = 0

        elif self.phase in ("lift", "transport", "release"):
            c = self.cubes[self.target]
            b = self.bins[self._cube_color(self.target)]
            self.grip = 1.0
            arm_z = CARRY_Z
            bw = self._bracelet_world()
            if self.phase == "lift":
                self.counter += 1
                if c["p"][2] > CLEAR_Z:
                    self.phase = "transport"
                    self.counter = 0
                elif bw[2] > CARRY_Z - 0.02 or (self.counter > 8 and c["p"][2] < self.grasp_cz + 0.01):
                    self.fail[self.target] = self.fail.get(self.target, 0) + 1
                    self.phase = "select"
                    self.grip = G_OPEN
            if self.phase == "transport":
                dm = self._delta_max(b[:2])
                bt = self._base_target(b[:2], np.pi + np.clip(wrap(self.carry_rot - np.pi), -dm, dm))
                base_t = bt
                self.counter += 1
                err = np.linalg.norm(bw[:2] - b[:2])
                if err < 0.012 and np.linalg.norm(self.basevel[:2]) < 0.05:
                    self.phase = "release"
                    self.counter = 0
                if c["p"][2] < 0.46 and self._which_bin(c["p"]) is None:  # dropped
                    self.phase = "select"
            if self.phase == "release":
                self.grip = G_OPEN
                self.counter += 1
                if self.counter >= 4:
                    self.phase = "select"

        # ---- arm command
        qd = self._arm_q(arm_z, gam_t)
        e = qd - self.q
        e[[0, 2, 4, 6]] = wrap(e[[0, 2, 4, 6]])
        small = np.abs(e) < 0.05
        self.qi = np.where(small, np.clip(self.qi + 0.25 * e, -0.08, 0.08), self.qi * 0.8)
        dq = K_ARM * e + self.qi
        m = np.max(np.abs(dq[:6]))
        if m > 0.1:
            dq[:6] *= 0.1 / m
        dq[6] = np.clip(dq[6], -0.1, 0.1)
        act[3:10] = dq
        # ---- base command
        if base_t is not None:
            d = base_t - self.base
            d[2] = wrap(d[2])
            d[:2] /= 0.87
            smallb = np.abs(d) < 0.03
            self.bi = np.where(smallb, np.clip(self.bi + 0.3 * d, -0.05, 0.05), 0.0)
            d = d + self.bi
            if base_t is not None and base_t[0] < BASE_XMIN:
                d[0] = max(d[0], BASE_XMIN - self.base[0])
            act[0:3] = np.clip(d, -0.1, 0.1)
        act[10] = self.grip
        return act
