"""State-machine pick-and-place approach for SortClutteredBlocks3D (variable count)."""
import re
import os as _os
import numpy as np

try:
    from kin import Kin, down_rot
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kin import Kin, down_rot

BIN_ORDER = ["bin_red", "bin_green", "bin_blue", "bin_yellow"]
ARM_KP = 2.3
ARM_KV = 0.14
BASE_TARGET = np.array([0.6, 0.0, np.pi])
HOME_ARM = np.array([0.0, -0.35, 3.14, -2.55, 0.0, -0.87, 1.57])
CUBE_H = 0.01  # half size
TRAVEL_Z = 0.60
PRE_DZ = 0.06
CLOSE_STEPS = 1
LIFT_RAMP = [0.1, 0.3, 0.6, 1.0]
OPEN_CMD = 0.62
RELEASE_CMD = 0.3
GRASP_DZ = 0.042
STACK_DZ = 0.0
PRE_RAMP = None
DESC_RAMP = [0.5, 1.0]
PRE_XY_TOL = 0.004
APP_DZ = 0.05
LIFT_Z = 0.52
MOVE_STEP = 0.1
BIN_CLEAR = 0.09
PUSH_THRESH = -0.003
PUSH_NDIR = 16
PUSH_LENS = (0.03, 0.045, 0.06)
PUSHER_HD = 0.011
PUSHER_HW = 0.012
PUSH_Z = 0.452
PRESS_DZ = 0.034
DRAG_YAW = 0.0
DRAG_BACK = 0.0
DRAG_GRIP = 1.0
PUSH_STEP = 0.02
DRAG_MOVE = 0.02
DRAG_LEN = 0.05
REL_TOL = 0.02
REL_VEL = 0.5
PAD_U0 = 0.0
PAD_U1 = 0.036
PAD_V = 0.012
GRASP_DYAW = (0.0, 0.2, -0.2, 0.35, -0.35)
GRASP_OFFS = (0.0, 0.003, -0.003, 0.006, -0.006)
CLOSE_CMD = 1.0
RELEASE_Z = 0.552
SAFE_Z = 0.56
SAFE_R = 0.045
SLOT_OFF = 0.011


def grasp_score(pxy, th, D, OY, bins):
    """Best grasp (key, phi, offset_vec, clearance) for a cube at pxy with yaw th."""
    dy = np.array(GRASP_DYAW)
    phis = np.concatenate([th + dy, th + np.pi / 2 + dy])
    dys = np.concatenate([dy, dy])
    offs = np.array(GRASP_OFFS)
    U = np.stack([np.cos(phis), np.sin(phis)], 1)
    V = np.stack([-U[:, 1], U[:, 0]], 1)
    score = np.full((len(phis), len(offs)), 1.0)
    if len(D):
        du = U @ D.T
        dv = (V @ D.T)[:, None, :] - offs[None, :, None]
        rel = OY[None, :] - phis[:, None]
        h = 0.01 * (np.abs(np.cos(rel)) + np.abs(np.sin(rel)))
        gv = np.abs(dv) - h[:, None, :] - PAD_V
        for sg in (-1.0, 1.0):
            gu = np.maximum(PAD_U0 - (sg * du + h), (sg * du - h) - PAD_U1)
            score = np.minimum(score, np.maximum(gu[:, None, :], gv).min(axis=2))
    GP = pxy[None, None, :2] + offs[None, :, None] * V[:, None, :]
    W = GP[:, :, None, :] + _BODY_PTS[None, None, :, 0:1] * U[:, None, None, :] + _BODY_PTS[None, None, :, 1:2] * V[:, None, None, :]
    for bp in bins:
        q = np.abs(W - bp)
        score = np.minimum(score, (np.maximum(q[..., 0], q[..., 1]) - 0.05).min(axis=2))
    key = np.minimum(score, 0.004) - 0.01 * np.abs(offs)[None, :] - 0.002 * np.abs(dys)[:, None]
    i, j = np.unravel_index(np.argmax(key), key.shape)
    return key[i, j], phis[i], offs[j] * V[i], score[i, j]


_BODY_PTS = np.array([(sg * r, t * 0.014) for sg in (-1.0, 1.0)
                      for t, r in ((-1.0, 0.033), (0.0, 0.033), (1.0, 0.033), (-1.0, 0.075), (0.0, 0.075), (1.0, 0.075), (0.0, 0.055))])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_from_quat(qw, qx, qy, qz):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.kin = Kin()

    # ------------------------------------------------------------------ utils
    def _robot(self, state):
        r = state.get_object_from_name("robot")
        b = np.array([state.get(r, f) for f in ["pos_base_x", "pos_base_y", "pos_base_rot"]])
        a = np.array([state.get(r, f"pos_arm_joint{i}") for i in range(1, 8)])
        g = state.get(r, "pos_gripper")
        self._vel = np.array([state.get(r, f"vel_arm_joint{i}") for i in range(1, 8)])
        return b, a, g

    def _pose(self, state, name):
        o = state.get_object_from_name(name)
        return np.array([state.get(o, f) for f in ["x", "y", "z", "qw", "qx", "qy", "qz"]])

    def _goal_ok(self, state, cname):
        c = self._pose(state, cname)
        b = self._pose(state, self.target_bin[cname])
        return (abs(c[0] - b[0]) < 0.028 and abs(c[1] - b[1]) < 0.028
                and b[2] + 0.016 < c[2] < b[2] + 0.088)

    # ------------------------------------------------------------------ api
    def reset(self, state, info):
        names = [n for n in state.get_object_names() if re.fullmatch(r"cube\d+", n)]
        names.sort(key=lambda n: int(n[4:]))
        self.cubes = names
        n = len(names)
        idx = sorted(int(c[4:]) for c in names)
        lo = idx[0] if idx else 1
        self.target_bin = {}
        for c in names:
            k = int(c[4:]) - lo
            self.target_bin[c] = BIN_ORDER[min(3, (k * 4) // max(n, 1))]
        self.bin_count = {b: 0 for b in BIN_ORDER}
        for c in names:
            if self._goal_ok(state, c):
                self.bin_count[self.target_bin[c]] += 1
        self.phase = "select"
        self.arm_target = None
        self.grip = OPEN_CMD
        self.cur = None
        self.steps_in_phase = 0
        self.fail = {c: 0 for c in names}
        self.t = 0
        self.pushes = {}
        self.n_drag = 0
        self.n_drag_ok = 0
        self.ref = []

    def _set_arm_target(self, base, pos, yaw, ramp=None, cramp=None, off=None):
        p_start = getattr(self, 'cmd_pos', None)
        goal = np.asarray(pos, float)
        if off is not None:
            pos = goal + np.array([off[0], off[1], 0.0])
        self.cmd_pos = np.asarray(pos, float)
        seed = self.arm_target if self.arm_target is not None else HOME_ARM
        q_start = None if self.arm_target is None else self.arm_target.copy()
        q, err = self.kin.ik(base, seed, np.asarray(pos, float), down_rot(yaw), iters=200)
        if err > 0.01:
            q2, err2 = self.kin.ik(base, HOME_ARM, np.asarray(pos, float), down_rot(yaw), iters=300)
            if err2 < err:
                q, err = q2, err2
        self.arm_target = q
        self.goal_pos = goal
        self.ref = []
        if ramp and q_start is not None:
            self.ref = [q_start + f * (q - q_start) for f in ramp]
        elif cramp and q_start is not None and p_start is not None:
            qs = q_start
            pos = np.asarray(pos, float)
            for f in cramp[:-1]:
                qs, _ = self.kin.ik(base, qs, p_start + f * (pos - p_start), down_rot(yaw), iters=50)
                self.ref.append(qs)

    def _ee(self, base, arm):
        p, _ = self.kin.fk(base, arm)
        return p

    def _choose_cube(self, state):
        cands = []
        pos = {c: self._pose(state, c) for c in self.cubes}
        bins = [self._pose(state, b)[:2] for b in BIN_ORDER]
        self._pos_cache, self._bins_cache = pos, bins
        self.movable = []
        for c in self.cubes:
            if self._goal_ok(state, c):
                continue
            if self.fail[c] >= 6:
                continue
            p = pos[c]
            if p[2] < 0.3:  # fell off the table
                continue
            on_wall = False
            if self._in_any_bin(state, p):
                if p[2] < 0.49:
                    continue
                on_wall = True  # resting on a bin wall: re-pick from above
            o = state.get_object_from_name(c)
            if abs(state.get(o, "vz")) > 0.05 or abs(state.get(o, "vx")) > 0.05 or abs(state.get(o, "vy")) > 0.05:
                continue
            self.movable.append(c)
            th = _yaw_from_quat(*p[3:7])
            D, OY = self._neighbours(pos, c, p)
            best = grasp_score(p[:2], th, D, OY, [] if on_wall else bins)
            pri = min(best[3], 0.008) + 0.2 * (p[2] - 0.41) - 0.004 * self.fail[c]
            cands.append((pri, c, best[1], best[2], best[3]))
        if not cands:
            return None
        cands.sort(key=lambda x: -x[0])
        self.last_cands = cands
        return cands[0][1], cands[0][2], cands[0][3]

    def _neighbours(self, pos, c, p, zmin=-0.012):
        D, OY = [], []
        for o in self.cubes:
            if o == c:
                continue
            d = pos[o][:3] - p[:3]
            if d[2] < zmin or d[2] > 0.06 or np.hypot(d[0], d[1]) > 0.12:
                continue
            D.append(d[:2])
            OY.append(_yaw_from_quat(*pos[o][3:7]))
        return np.array(D).reshape(-1, 2), np.array(OY)

    def _plan_push(self, state):
        """Find a cube to drag (press on top + slide) so that it becomes graspable."""
        pos, bins = self._pos_cache, self._bins_cache
        best = None
        dirs = np.linspace(0, 2 * np.pi, PUSH_NDIR, endpoint=False)
        for c in self.movable:
            p = pos[c]
            if p[2] > 0.416 or self.pushes.get(c, 0) >= 2:
                continue
            th = _yaw_from_quat(*p[3:7])
            D, OY = self._neighbours(pos, c, p, zmin=-0.03)
            if not len(D):
                continue
            for ang in dirs:
                dvec = np.array([np.cos(ang), np.sin(ang)])
                perp = np.array([-dvec[1], dvec[0]])
                rel_c = th - ang
                hc = 0.01 * (abs(np.cos(rel_c)) + abs(np.sin(rel_c)))
                a_ = D @ dvec
                b_ = np.abs(D @ perp)
                relo = OY - ang
                ho = 0.01 * (np.abs(np.cos(relo)) + np.abs(np.sin(relo)))
                L = DRAG_MOVE
                dest = p[:2] + dvec * L
                if np.linalg.norm(dest) > 0.17:
                    continue
                if min(max(abs(dest[0] - bp[0]), abs(dest[1] - bp[1])) for bp in bins) < 0.05 + 0.03:
                    continue
                # swept corridor of the dragged cube (front side) must be free
                hit = (b_ < hc + ho - 0.001) & (a_ > 0) & (a_ < L + hc + ho + 0.003)
                if hit.any():
                    continue
                sc = grasp_score(dest, th, D - dvec * L, OY, bins)[3]
                if sc < -0.001:
                    continue
                key = min(sc, 0.006)
                if best is None or key > best[0]:
                    best = (key, c, dvec, L, 0.0)
        return best

    def _grasp_yaw(self, phi, cur_yaw):
        # fingers axis = site x = (cos yaw, sin yaw) -> yaw = phi (mod pi)
        y0 = _wrap(phi)
        y1 = _wrap(y0 + np.pi)
        return y0 if abs(_wrap(y0 - cur_yaw)) <= abs(_wrap(y1 - cur_yaw)) else y1

    def _slot(self, state, cname):
        b = self._pose(state, self.target_bin[cname])
        k = self.bin_count[self.target_bin[cname]] % 5
        offs = [(SLOT_OFF, SLOT_OFF), (-SLOT_OFF, -SLOT_OFF), (SLOT_OFF, -SLOT_OFF),
                (-SLOT_OFF, SLOT_OFF), (0.0, 0.0)]
        return np.array([b[0] + offs[k][0], b[1] + offs[k][1]])

    def _held(self, cp, ee, tol=0.02):
        dz = ee[2] - cp[2]
        return np.linalg.norm(cp[:2] - ee[:2]) < tol and 0.01 < dz < 0.055

    def _in_any_bin(self, state, p):
        for b in BIN_ORDER:
            bp = self._pose(state, b)
            if abs(p[0] - bp[0]) < 0.052 and abs(p[1] - bp[1]) < 0.052:
                return True
        return False

    def _clear_of_bins(self, state, ee, margin):
        for b in BIN_ORDER:
            bp = self._pose(state, b)
            if abs(ee[0] - bp[0]) < 0.05 + margin and abs(ee[1] - bp[1]) < 0.05 + margin:
                return False
        return True

    def _goto(self, phase):
        self.phase = phase
        self.steps_in_phase = 0

    def _set_grasp(self, p):
        self.grasp_xy = p[:2] + self.goff
        self.grasp_z = max(p[2], 0.41) + GRASP_DZ + (STACK_DZ if p[2] > 0.416 else 0.0)

    def _line(self, a, b, step):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        return [a + (b - a) * (k / n) for k in range(1, n + 1)]

    def _seg(self, a, b, mode):
        """Line a->b, but if it skims a bin below wall height, go via TRAVEL_Z (desc: descend last, rise: rise first)."""
        a = np.asarray(a, float); b = np.asarray(b, float)
        bins = getattr(self, "_bins_cache", None) or []
        unsafe = False
        for t in np.linspace(0.0, 1.0, 21):
            q = a + (b - a) * t
            if q[2] >= SAFE_Z:
                continue
            if mode == "desc" and np.linalg.norm(q[:2] - b[:2]) < 0.015:
                continue
            if mode == "rise" and np.linalg.norm(q[:2] - a[:2]) < 0.015:
                continue
            for bp in bins:
                dx = max(abs(q[0] - bp[0]) - 0.05, 0.0)
                dy = max(abs(q[1] - bp[1]) - 0.05, 0.0)
                if np.hypot(dx, dy) < SAFE_R:
                    unsafe = True
                    break
            if unsafe:
                break
        if not unsafe:
            return self._line(a, b, MOVE_STEP)
        if mode == "desc":
            m = np.array([b[0], b[1], max(TRAVEL_Z, a[2])])
        else:
            m = np.array([a[0], a[1], max(TRAVEL_Z, b[2])])
        return self._line(a, m, MOVE_STEP) + self._line(m, b, MOVE_STEP)

    def _set_path(self, base, wps, yaw):
        """Follow Cartesian waypoints (one per control step); final waypoint is the target."""
        q = self.arm_target.copy() if self.arm_target is not None else np.array(HOME_ARM, float)
        self._set_arm_target(base, wps[-1], yaw)
        ref = []
        for w in wps[:-1]:
            q, _ = self.kin.ik(base, q, np.asarray(w, float), down_rot(yaw), iters=60)
            ref.append(q)
        self.ref = ref

    def _do_select(self, state, cur_yaw):
        self.phase = "select"
        self.steps_in_phase = 0
        ch = self._choose_cube(state)
        if ch is None:
            self.cur = None
            if self.arm_target is None:
                self._set_arm_target(BASE_TARGET, [0.3, 0.0, TRAVEL_Z], np.pi / 2)
            return
        if self.last_cands[0][4] < PUSH_THRESH:
            plan = self._plan_push(state)
            if plan is not None:
                self._start_push(state, plan, cur_yaw)
                return
        self.cur, phi, self.goff = ch
        self.xy_off = np.zeros(2)
        self.n_corr = 0
        self.retarget = 0
        p = self._pose(state, self.cur)
        self.yaw = self._grasp_yaw(phi, cur_yaw)
        self._set_grasp(p)
        self.grip = OPEN_CMD
        tgt = np.array([self.grasp_xy[0], self.grasp_xy[1], self.grasp_z + APP_DZ])
        self._set_path(BASE_TARGET, self._path_from_cmd(state, tgt), self.yaw)
        self._goto("approach")

    def _path_from_cmd(self, state, tgt):
        start = getattr(self, "cmd_pos", None)
        if start is None or self.arm_target is None:
            return [np.array([tgt[0], tgt[1], TRAVEL_Z]), tgt]
        wps = []
        d = tgt[:2] - start[:2]
        dist = np.linalg.norm(d)
        if not self._clear_of_bins(state, start, 0.04):
            if dist > BIN_CLEAR + 0.02:
                w = np.array([start[0] + d[0] / dist * BIN_CLEAR, start[1] + d[1] / dist * BIN_CLEAR, max(TRAVEL_Z, start[2])])
            else:
                w = np.array([tgt[0], tgt[1], max(TRAVEL_Z, start[2])])
            wps = self._line(start, w, MOVE_STEP)
            start = w
        return wps + self._seg(start, tgt, 'desc')

    def _start_push(self, state, plan, cur_yaw):
        _, c, dvec, L, back = plan
        self.pushes[c] = self.pushes.get(c, 0) + 1
        self.cur = c
        p = self._pos_cache[c]
        self.push_start = p[:2] - dvec * DRAG_BACK
        self.push_end = p[:2] + dvec * DRAG_LEN
        self.drag_c0 = p[:2].copy()
        perp = np.arctan2(dvec[0], -dvec[1]) + DRAG_YAW
        self.yaw = self._grasp_yaw(perp, cur_yaw)
        self.grip = DRAG_GRIP
        self.xy_off = np.zeros(2)
        self.n_corr = 0
        self.press_z = p[2] + PRESS_DZ
        tgt = np.array([self.push_start[0], self.push_start[1], PUSH_Z + 0.05])
        self._set_path(BASE_TARGET, self._path_from_cmd(state, tgt), self.yaw)
        self._goto("papp")

    def get_action(self, state):
        self.t += 1
        base, arm, grip = self._robot(state)
        vel = self._vel
        self.steps_in_phase += 1
        ee = self._ee(base, arm)
        _, Rc = self.kin.fk(base, arm)
        cur_yaw = np.arctan2(Rc[1, 0], Rc[0, 0])

        if self.phase == "select":
            self._do_select(state, cur_yaw)
        elif self.phase == "approach":
            e = ee - self.goal_pos
            settled = not self.ref and np.abs(vel).max() < 0.08
            if self.steps_in_phase > 20:
                self.fail[self.cur] += 1
                self._do_select(state, cur_yaw)
            elif settled and self.steps_in_phase >= 2 and np.linalg.norm(e[:2]) >= PRE_XY_TOL and self.n_corr < 4:
                # steady-state offset: shift the commanded target to compensate
                self.n_corr += 1
                self.xy_off = self.xy_off - e[:2]
                self._set_arm_target(BASE_TARGET, self.goal_pos, self.yaw, off=self.xy_off)
            elif not self.ref and np.linalg.norm(e[:2]) < PRE_XY_TOL and abs(e[2]) < 0.01 or (settled and self.n_corr >= 4):
                p = self._pose(state, self.cur)
                if np.linalg.norm(p[:2] + self.goff - self.grasp_xy) > 0.006 and self.retarget < 2:
                    # cube moved: re-target above it
                    self.retarget += 1
                    self._set_grasp(p)
                    self._set_arm_target(BASE_TARGET, [self.grasp_xy[0], self.grasp_xy[1], self.grasp_z + APP_DZ], self.yaw, off=self.xy_off)
                    self.steps_in_phase = 0
                else:
                    self._set_arm_target(BASE_TARGET, [self.grasp_xy[0], self.grasp_xy[1], self.grasp_z], self.yaw, cramp=DESC_RAMP, off=self.xy_off)
                    self._goto("descend")
        elif self.phase == "papp":
            e = ee - self.goal_pos
            settled = not self.ref and np.abs(vel).max() < 0.08
            if settled and self.steps_in_phase >= 2 and np.linalg.norm(e[:2]) >= 0.004 and self.n_corr < 3:
                self.n_corr += 1
                self.xy_off = self.xy_off - e[:2]
                self._set_arm_target(BASE_TARGET, self.goal_pos, self.yaw, off=self.xy_off)
            elif (not self.ref and np.linalg.norm(e[:2]) < 0.004 and abs(e[2]) < 0.01) or (settled and self.n_corr >= 3) or self.steps_in_phase > 15:
                self._set_arm_target(BASE_TARGET, [self.push_start[0], self.push_start[1], self.press_z], self.yaw, cramp=DESC_RAMP, off=self.xy_off)
                self._goto("pdown")
        elif self.phase == "pdown":
            if (not self.ref and (abs(ee[2] - self.goal_pos[2]) < 0.006 or (self.steps_in_phase >= 3 and np.abs(vel).max() < 0.05))) or self.steps_in_phase > 5:
                a = np.array([self.push_start[0], self.push_start[1], self.press_z])
                b = np.array([self.push_end[0], self.push_end[1], self.press_z])
                o = np.array([self.xy_off[0], self.xy_off[1], 0.0])
                self._set_path(BASE_TARGET, [w + o for w in self._line(a, b, PUSH_STEP)], self.yaw)
                self.goal_pos = b
                self._goto("pmove")
        elif self.phase == "pmove":
            if (not self.ref and np.linalg.norm(ee[:2] - self.goal_pos[:2]) < 0.006) or self.steps_in_phase > 8:
                self._set_arm_target(BASE_TARGET, [ee[0], ee[1], PUSH_Z + 0.05], self.yaw)
                self._goto("pup")
        elif self.phase == "pup":
            if ee[2] > PUSH_Z + 0.03 or self.steps_in_phase > 4:
                cp = self._pose(state, self.cur)
                self.n_drag += 1
                if np.linalg.norm(cp[:2] - self.drag_c0) < 0.004:
                    self.pushes[self.cur] = 99  # drag ineffective: do not retry this cube
                else:
                    self.n_drag_ok += 1
                self._do_select(state, cur_yaw)
        elif self.phase == "descend":
            if (not self.ref and np.linalg.norm(ee - self.goal_pos) < 0.006) or self.steps_in_phase > 8:
                self.grip = CLOSE_CMD
                self._goto("close")
        elif self.phase == "close":
            if self.steps_in_phase >= CLOSE_STEPS:
                sl = self._slot(state, self.cur)
                p0 = np.array([ee[0], ee[1], self.cmd_pos[2]])
                wps = [p0 + [0, 0, 0.03], np.array([ee[0], ee[1], LIFT_Z])]
                d = sl - ee[:2]
                dist = np.linalg.norm(d)
                if dist > BIN_CLEAR + 0.02:
                    w = np.array([sl[0] - d[0] / dist * BIN_CLEAR, sl[1] - d[1] / dist * BIN_CLEAR, TRAVEL_Z])
                    wps += self._seg(wps[-1], w, 'rise')
                else:
                    wps.append(np.array([ee[0], ee[1], TRAVEL_Z]))
                wps += self._line(wps[-1], np.array([sl[0], sl[1], RELEASE_Z]), MOVE_STEP)
                self._set_path(BASE_TARGET, wps, self.yaw)
                self._goto("carry")
        elif self.phase == "carry":
            cp = self._pose(state, self.cur)
            if self.steps_in_phase >= 3 and not self._held(cp, ee, 0.03):
                self.fail[self.cur] += 1
                self.grip = OPEN_CMD
                self._do_select(state, cur_yaw)
            elif self.steps_in_phase == 3 and len(self.ref) >= 1:
                # compensate the cube's offset in the gripper so the cube (not the ee) lands on the slot
                off = cp[:2] - ee[:2]
                if np.linalg.norm(off) > 0.003:
                    g = self.goal_pos.copy()
                    ref = self.ref
                    self._set_arm_target(BASE_TARGET, [g[0] - off[0], g[1] - off[1], g[2]], self.yaw)
                    self.ref = ref
            elif (not self.ref and np.linalg.norm(ee[:2] - self.goal_pos[:2]) < REL_TOL and np.abs(vel).max() < REL_VEL) or self.steps_in_phase > 25:
                self.grip = RELEASE_CMD
                self._goto("release")
        elif self.phase == "release":
            cp = self._pose(state, self.cur)
            if (ee[2] - cp[2] > 0.065) or self.steps_in_phase >= 5:
                self.bin_count[self.target_bin[self.cur]] += 1
                self._do_select(state, cur_yaw)

        return self._act(base, arm, vel)

    def _act(self, base, arm, vel):
        act = np.zeros(11, dtype=np.float32)
        db = BASE_TARGET - base
        db[2] = _wrap(db[2])
        act[:3] = np.clip(db, -0.3, 0.3)
        if self.arm_target is not None:
            cmd = self.ref.pop(0) if self.ref else self.arm_target
            act[3:10] = ARM_KP * (cmd - arm) + ARM_KV * vel
        act[10] = self.grip
        return act
