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
STAGE_R = 0.10          # staging distance from bin centre while still lifting
ROT_W = 2.0             # selection cost per rad of finger rotation
HOVER_DZ = 0.26         # bracelet height above cube while aligning
AMAX = 0.0              # max shoulder yaw used for finger rotation
DMAX = 0.0              # max base yaw deviation from pi
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
        a = self._alpha()
        off = rotz2(rot) @ np.array([MOUNT[0] + REACH * np.cos(a), -REACH * np.sin(a)])
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
        self.pre_gam = None
        self.pair = None
        self.pair_moves = 0
        self.sel_clr = 0.0
        self.alpha_t = 0.0
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
        """Total finger yaw offset: shoulder yaw (j1) + wrist yaw (j7)."""
        qc = self._arm_q(CARRY_Z)
        return wrap((self.q[0] - qc[0]) + wrap(self.q[6] - qc[6]))

    def _alpha(self):
        return wrap(self.q[0] - self._arm_q(CARRY_Z)[0])

    def _plan_alpha(self, total):
        """Split a total yaw target between shoulder (j1) and wrist (j7)."""
        a0 = self._alpha()
        d = wrap(total - self._gamma())
        self.alpha_t = float(np.clip(a0 + d / 2, -AMAX, AMAX))

    def _arm_q2(self, z, total):
        q = self._arm_q(z, wrap(total - self.alpha_t))
        q[0] += self.alpha_t
        return q

    def _select(self, exclude=None):
        best, bestv = None, None
        cur = self._bracelet_world()
        gam = self._gamma()
        for n in self.cube_names:
            if n == exclude or self._in_bin(n):
                continue
            c = self.cubes[n]
            yaw, tilt = cube_face_yaw(c["quat"])
            # topmost cubes: anything resting on it?
            covered = any(o["p"][2] > c["p"][2] + 0.012 and np.linalg.norm(o["p"][:2] - c["p"][:2]) < 0.022
                          for m, o in self.cubes.items() if m != n)
            dmax = self._delta_max(c["p"][:2])
            wrong = self._which_bin(c["p"]) is not None
            for k in range(2):
                phi = yaw + k * np.pi / 2
                clr = self._axis_clearance(n, phi)
                # base yaw delta needed with current gamma
                need = wrap_half(phi - np.pi / 2 - gam - np.pi)
                rot_cost = max(0.0, abs(need) - dmax)
                v = ((3.0 - 300 * clr if clr < -0.002 else 0.0) - c["p"][2] * 5 + (1.0 if covered else 0.0)
                     + 0.2 * np.linalg.norm(c["p"][:2] - cur[:2]) + ROT_W * rot_cost
                     + 0.5 * self.fail.get(n, 0) + 1.0 * self.fail.get((n, k), 0) + 0.2 * tilt
                     + (2.0 + 3.0 * self.fail.get(n, 0) if wrong else 0.0))
                if bestv is None or v < bestv:
                    best, bestv = (n, phi), v
                    self.sel_k = k
                    self.sel_clr = clr
        return best

    def _pair_plan(self):
        """For packed clusters: find two touching cubes to grab together and slide aside."""
        free = [n for n in self.cube_names if self._which_bin(self.cubes[n]["p"]) is None]
        if len(free) < 3:
            return None
        cen = np.mean([self.cubes[n]["p"][:2] for n in free], axis=0)
        best, bv = None, None
        for n in free:
            cn = self.cubes[n]["p"]
            yaw, tilt = cube_face_yaw(self.cubes[n]["quat"])
            if tilt > 0.2:
                continue
            for k in range(2):
                phi = yaw + k * np.pi / 2
                u = np.array([np.cos(phi), np.sin(phi)])
                v = np.array([-u[1], u[0]])
                for m in free:
                    if m == n:
                        continue
                    d = self.cubes[m]["p"] - cn
                    if abs(d[2]) > 0.008 or abs(d[:2] @ v) > 0.018 or not (0.012 < d[:2] @ u < 0.035):
                        continue
                    pc = (cn[:2] + self.cubes[m]["p"][:2]) / 2
                    clr = 1.0
                    for o in free:
                        if o in (n, m):
                            continue
                        e = self.cubes[o]["p"] - np.append(pc, cn[2])
                        if e[2] < -0.015 or e[2] > 0.03:
                            continue
                        al, pe = abs(e[:2] @ u), abs(e[:2] @ v)
                        clr = min(clr, max(al - 0.066, pe - 0.022))
                    if clr < -0.003:
                        continue
                    w = v if (pc - cen) @ v >= 0 else -v
                    dest = pc + 0.05 * w
                    if any(np.max(np.abs(dest - b[:2])) < 0.10 for b in self.bins.values()):
                        w = -w
                        dest = pc + 0.05 * w
                        if any(np.max(np.abs(dest - b[:2])) < 0.10 for b in self.bins.values()):
                            continue
                    val = -clr + self.fail.get(("pair", n, m), 0)
                    if bv is None or val < bv:
                        best, bv = (n, m, phi, w), val
        return best

    def _pick_p(self):
        if self.pair is not None:
            return (self.cubes[self.pair[0]]["p"] + self.cubes[self.pair[1]]["p"]) / 2
        return self.cubes[self.target]["p"]

    def _mark_fail(self):
        self.fail[self.target] = self.fail.get(self.target, 0) + 1
        key = (self.target, self.sel_k)
        self.fail[key] = self.fail.get(key, 0) + 1

    def _yaw_plan(self, txy):
        """Fix a gamma target at selection time; base rot = pi + remaining yaw."""
        dmax = self._delta_max(txy)
        gam = self._gamma()
        need = wrap_half(self.phi - np.pi / 2 - gam - np.pi)
        d = float(np.clip(need, -dmax, dmax))
        self.gam_target = wrap(gam + (need - d))
        self._plan_alpha(self.gam_target)

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
                self.pair = None
                self.pre_gam = None
                if (self.sel_clr < -0.002 or self.fail.get(self.target, 0) >= 2) and self.pair_moves < 12:
                    pp = self._pair_plan()
                    if pp is not None:
                        self.pair = pp
                        self.target, self.phi = pp[0], pp[2]
                        self.pair_moves += 1
                self._yaw_plan(self._pick_p()[:2])
                self.phase = "goto_pick"
                self.counter = 0

        if self.phase == "done":
            base_t = self.base.copy()
            self.grip = 0.0

        elif self.phase in ("goto_pick", "descend", "close"):
            c = {"p": self._pick_p()}
            rot, gam_t = self._base_rot(), self.gam_target
            base_t = self._base_target(c["p"][:2], rot)
            bw = self._bracelet_world()
            xy_err = np.linalg.norm(bw[:2] - c["p"][:2])
            if xy_err < 0.03 and abs(wrap(rot - self.base[2])) < 0.02:
                # close the loop on the actual bracelet position
                base_t[:2] = self.base[:2] + (c["p"][:2] - bw[:2])
            yaw_err = max(abs(wrap(base_t[2] - self.base[2])), abs(wrap(gam_t - self._gamma())))
            if self.phase == "goto_pick":
                self.grip = G_OPEN if self.pair is None else 0.0
                near_bin = any(np.max(np.abs(bw[:2] - b[:2])) < 0.085 for b in self.bins.values())
                if not near_bin:
                    arm_z = max(c["p"][2] + HOVER_DZ, 0.62)
                if xy_err < 0.02 and yaw_err < 0.25:
                    self.phase = "descend"
                    self.counter = 0
                    self.stall = 0
            if self.phase == "descend":
                gz = c["p"][2] + GRASP_DZ
                # stay a bit above until well aligned
                ok = xy_err < 0.006 and yaw_err < 0.04
                arm_z = gz if ok else max(gz, c["p"][2] + HOVER_DZ)
                self.counter += 1
                if bw[2] < gz + 0.004 and ok:
                    self.phase = "close"
                    self.counter = 0
                elif self.counter > 45:
                    self._mark_fail()
                    self.phase = "select"
                    self.counter = 0
            if self.phase == "close":
                if self.counter == 0:
                    self.grasp_z = c["p"][2] + GRASP_DZ
                    self.grasp_cz = c["p"][2]
                    self.carry_rot = self._base_rot()
                    self.others_z = {m: o["p"][2] for m, o in self.cubes.items() if m != self.target}
                arm_z = self.grasp_z
                base_t = None  # hold base still while closing
                self.grip = 1.0
                self.counter += 1
                if self.counter >= 5:
                    self.phase = "lift" if self.pair is None else "pmove"
                    self.counter = 0
                    if self.pair is not None:
                        self.pdest = c["p"][:2] + 0.05 * self.pair[3]
                        self.opened = 0

        elif self.phase in ("pmove", "plower", "popen"):
            bw = self._bracelet_world()
            self.counter += 1
            self.grip = 1.0
            if self.phase == "pmove":
                arm_z = self.grasp_z + 0.03
                if bw[2] > self.grasp_z + 0.02 or self.counter > 12:
                    base_t = self._base_target(self.pdest, self._base_rot())
                    if np.linalg.norm(bw[:2] - self.pdest) < 0.008 or self.counter > 40:
                        self.phase = "plower"
                        self.counter = 0
                else:
                    base_t = None
            if self.phase == "plower":
                arm_z = self.grasp_z + 0.004
                if bw[2] < self.grasp_z + 0.01 or self.counter > 15:
                    self.phase = "popen"
                    self.counter = 0
            if self.phase == "popen":
                arm_z = self.grasp_z + 0.004 if self.counter < 5 else CARRY_Z
                self.grip = 0.0
                if self.counter >= 6:
                    self.fail[("pair", self.pair[0], self.pair[1])] = self.fail.get(("pair", self.pair[0], self.pair[1]), 0) + 1
                    self.pair = None
                    self.phase = "select"

        elif self.phase in ("lift", "transport", "release", "putback"):
            c = self.cubes[self.target]
            b = self.bins[self._cube_color(self.target)]
            self.grip = 1.0
            arm_z = CARRY_Z
            bw = self._bracelet_world()
            if self.phase == "lift":
                self.counter += 1
                col = self._cube_color(self.target)
                co = [m for m, z0 in self.others_z.items()
                      if self.cubes[m]["p"][2] > z0 + 0.025 and self._which_bin(self.cubes[m]["p"]) is None
                      and self._cube_color(m) != col]
                if co:
                    self._mark_fail()
                    self.phase = "putback"
                    self.counter = 0
                    self.opened = 0
                elif c["p"][2] > CLEAR_Z:
                    self.phase = "transport"
                    self.counter = 0
                elif c["p"][2] > self.grasp_cz + 0.045 and self.counter > 2:
                    # clear of the pile: start moving to a staging point just outside the bin
                    dv = bw[:2] - b[:2]
                    dn = np.linalg.norm(dv)
                    if dn > STAGE_R:
                        base_t = self._base_target(b[:2] + dv / dn * STAGE_R, np.pi)
                elif bw[2] > CARRY_Z - 0.02 or (self.counter > 8 and c["p"][2] < self.grasp_cz + 0.01):
                    self._mark_fail()
                    self.phase = "select"
                    self.grip = G_OPEN
            if self.phase in ("lift", "transport", "release"):
                if self.phase == "lift" and self.counter == 3:
                    nxt = self._select(exclude=self.target)
                    if nxt is not None:
                        g0 = self._gamma()
                        self.pre_gam = wrap(g0 + wrap_half(nxt[1] - np.pi / 2 - g0 - np.pi))
                        self._plan_alpha(self.pre_gam)
                    else:
                        self.pre_gam = None
                if self.pre_gam is not None:
                    gam_t = self.pre_gam
            if self.phase == "transport":
                dm = self._delta_max(b[:2])
                bt = self._base_target(b[:2], np.pi)
                base_t = bt
                self.counter += 1
                err = np.linalg.norm(bw[:2] - b[:2])
                if err < min(0.03, 0.012 + 0.0005 * max(0, self.counter - 30)) and np.linalg.norm(self.basevel[:2]) < 0.05:
                    self.phase = "release"
                    self.counter = 0
                held = np.linalg.norm(c["p"] - (bw - np.array([0, 0, HOLD_TOOL]))) < 0.05
                if not held and self._which_bin(c["p"]) is None:  # dropped
                    self.phase = "select"
            if self.phase == "putback":
                arm_z = self.grasp_z + 0.01
                self.counter += 1
                if bw[2] < self.grasp_z + 0.02 or self.counter > 20:
                    self.opened = getattr(self, "opened", 0) + 1
                    self.grip = G_OPEN
                    if self.opened >= 5:
                        self.opened = 0
                        self.phase = "select"
            if self.phase == "release":
                self.grip = G_OPEN
                self.counter += 1
                if self.counter >= 3:
                    self.phase = "select"

        # ---- arm command
        qd = self._arm_q2(arm_z, gam_t)
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
