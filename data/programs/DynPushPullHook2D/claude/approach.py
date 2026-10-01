"""Approach for DynPushPullHook2DEnv."""
import numpy as np

TWO_PI = 2.0 * np.pi


def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


class GeneratedApproach:
    # geometry constants (calibrated empirically)
    GRASP_OFF = 0.07          # grasp point distance beyond arm_joint
    PLATE_OFF = 0.043         # gripper base plate front beyond arm_joint
    EFF_R = 0.27              # effective collision radius of the robot base
    XMIN, XMAX = 0.245, 3.255
    YMIN, YMAX = 0.245, 1.483

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.lo = np.array([-0.05, -0.05, -0.06544985, -0.1, -0.02])
        self.hi = -self.lo

    # ---------- state helpers ----------
    def _obj(self, state, name):
        return state.get_object_from_name(name)

    def g(self, state, name, feat):
        return float(state.get(self._obj(state, name), feat))

    def robot(self, state):
        return (self.g(state, 'robot', 'x'), self.g(state, 'robot', 'y'),
                self.g(state, 'robot', 'theta'), self.g(state, 'robot', 'arm_joint'),
                self.g(state, 'robot', 'finger_gap'))

    def hook_frame(self, state):
        """Return (origin, u, n, theta) of the hook. u = local +x (free end -> corner)."""
        hx = self.g(state, 'hook', 'x')
        hy = self.g(state, 'hook', 'y')
        th = self.g(state, 'hook', 'theta')
        u = np.array([np.cos(th), np.sin(th)])
        n = np.array([-np.sin(th), np.cos(th)])
        return np.array([hx, hy]), u, n, th

    def hook_dist(self, state, p):
        o, u, n, th = self.hook_frame(state)
        hw = 0.5 * self.bar_w
        segs = [(o - hw * n, o - self.bar_len * u - hw * n),
                (o - hw * u, o - hw * u - self.stub_len * n)]
        best = 1e9
        p = np.asarray(p, dtype=float)
        for a, b in segs:
            d = b - a
            L2 = float(d.dot(d))
            t = 0.0 if L2 < 1e-9 else float(np.clip((p - a).dot(d) / L2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (a + t * d))))
        return best - hw

    def free_end(self, state):
        """Centre of the free end face of the long bar."""
        o, u, n, th = self.hook_frame(state)
        return o - self.bar_len * u - 0.5 * self.bar_w * n

    def blocks(self, state):
        out = []
        for name in state.get_object_names():
            if name.startswith('obstruction') or name == 'target_block':
                out.append(name)
        return out

    def rect_extent(self, state, name):
        w = self.g(state, name, 'width') / 2.0
        h = self.g(state, name, 'height') / 2.0
        t = self.g(state, name, 'theta')
        ex = abs(w * np.cos(t)) + abs(h * np.sin(t))
        ey = abs(w * np.sin(t)) + abs(h * np.cos(t))
        return ex, ey

    # ---------- action helpers ----------
    def act(self, dx=0.0, dy=0.0, dth=0.0, da=0.0, dg=0.0):
        a = np.array([dx, dy, dth, da, dg], dtype=np.float64)
        a = np.clip(a, self.lo * 0.99, self.hi * 0.99)
        return a.astype(np.float32)

    def move_to(self, state, tx, ty, tol=0.004, scale=1.0, extra=None):
        """Return (action, done). Treats a stalled base as done."""
        x, y = self.g(state, 'robot', 'x'), self.g(state, 'robot', 'y')
        tx = float(np.clip(tx, self.XMIN, self.XMAX))
        ty = float(np.clip(ty, self.YMIN, self.YMAX))
        dx, dy = tx - x, ty - y
        done = (abs(dx) < tol and abs(dy) < tol) or self.stall >= 4
        kw = extra or {}
        return self.act(dx=dx * scale, dy=dy * scale, **kw), done

    # ---------- lifecycle ----------
    def reset(self, state, info):
        self.bar_len = self.g(state, 'hook', 'length_side1')
        self.bar_w = self.g(state, 'hook', 'width')
        self.stub_len = self.g(state, 'hook', 'length_side2')
        self.phase = 'assess'
        self.t = 0
        self.stall = 0
        self._pose = None
        self.var = 0
        self._var_hits = 0
        self.stuck = 0
        self.rel = None
        self.rot_target = None
        self.push_target = None
        self.d_grasp = 0.545
        self.substep = 0
        self.last_block = None
        self.replans = 0

    # ---------- geometry planning ----------
    def grasp_base(self, state, d):
        o, u, n, th = self.hook_frame(state)
        fe = self.free_end(state)
        return fe - d * u

    def _inb(self, p, m=0.004):
        return (self.XMIN + m <= p[0] <= self.XMAX - m and
                self.YMIN + m <= p[1] <= self.YMAX - m)

    def feasible_d(self, state):
        """Largest standoff d with in-bounds base pose AND staging pose."""
        o, u, n, th = self.hook_frame(state)
        for d in np.arange(0.545, 0.305, -0.005):
            b = self.grasp_base(state, d)
            if self._inb(b) and self._inb(b - self.stage_gap(d) * u):
                return float(d)
        return None

    def stage_gap(self, d):
        return max(0.2, 0.52 - d)

    # ---------- main ----------
    def get_action(self, state):
        self.t += 1
        pose = np.array([self.g(state, 'robot', 'x'), self.g(state, 'robot', 'y'),
                         self.g(state, 'robot', 'theta'), self.g(state, 'robot', 'arm_joint'),
                         self.g(state, 'robot', 'finger_gap')])
        prev = getattr(self, '_pose', None)
        moved = prev is None or np.abs(pose - prev).max() >= 0.0015
        if moved:
            self.stall = 0
            if self._var_hits > 6:
                self.var = 0
                self._var_hits = 0
            else:
                self._var_hits += 1
        else:
            self.stall += 1
            self.var = (self.var + 1) % 6
            self._var_hits = 0
        self._pose = pose
        phase0 = self.phase
        a = getattr(self, 'ph_' + self.phase)(state)
        if self.phase != phase0:
            self.var = 0
            self._var_hits = 0
            return a
        return self._apply_variant(a)

    def _apply_variant(self, a):
        v = self.var
        if v == 0:
            return a
        a = np.array(a, dtype=np.float64)
        if v == 1:
            a[0] = 0.0
        elif v == 2:
            a[1] = 0.0
        elif v == 3:
            a[0] = 0.0
            a[1] = 0.0
        elif v == 4:
            a[[0, 1, 3, 4]] = 0.0
        elif v == 5:
            a[[0, 1, 2, 4]] = 0.0
        return a.astype(np.float32)



    # ----- assess -----
    def ph_assess(self, state):
        d = self.feasible_d(state)
        if d is None:
            self.phase = 'push_prep'
            self.substep = 0
            return self.ph_push_prep(state)
        self.d_grasp = d
        self.phase = 'grasp_prep'
        self.substep = 0
        return self.ph_grasp_prep(state)

    # ----- pushing the hook to a reachable spot -----
    def com_local(self):
        a1 = self.bar_len * self.bar_w
        a2 = self.bar_w * max(self.stub_len - self.bar_w, 1e-6)
        cx = (a1 * (-self.bar_len / 2) + a2 * (-self.bar_w / 2)) / (a1 + a2)
        cy = (a1 * (-self.bar_w / 2) + a2 * (-(self.bar_w + self.stub_len) / 2)) / (a1 + a2)
        return cx, cy

    def push_geom(self, state, mode):
        """Return (standoff, push direction) in world coords."""
        o, u, n, th = self.hook_frame(state)
        if mode == 'right':
            lat = -min(self.stub_len - 0.06, self.bar_w + self.EFF_R + 0.12)
            p = o + (-self.bar_w - self.EFF_R - 0.005) * u + lat * n
            return p, u
        # mode 'cw': push up on the underside of the bar, left of the COM
        best = None
        for lx in np.arange(self.bar_len - 0.06, 0.55, -0.05):
            p = o - lx * u - (self.bar_w + self.EFF_R + 0.005) * n
            if self._inb(p, 0.01) and self._inb(p - 0.25 * n, 0.01):
                best = p
                break
        if best is None:
            best = o - 0.8 * u - (self.bar_w + self.EFF_R + 0.005) * n
        return best, n

    def push_mode(self, state):
        hth = self.g(state, 'hook', 'theta')
        o = self.hook_frame(state)[0]
        if hth > 0.10:
            return 'cw'
        return 'right'

    def ph_push_prep(self, state):
        self.pmode = self.push_mode(state)
        self.pstage = 0
        self.phase = 'push'
        return self.ph_push(state)

    def ph_push(self, state):
        d = self.feasible_d(state)
        if d is not None:
            self.d_grasp = d
            self.phase = 'grasp_prep'
            self.substep = 0
            return self.ph_grasp_prep(state)
        if self.t > 620:
            self.d_grasp = 0.31
            self.phase = 'grasp_prep'
            self.substep = 0
            return self.ph_grasp_prep(state)
        mode = self.push_mode(state)
        if mode != getattr(self, 'pmode', None):
            self.pmode = mode
            self.pstage = 0
        standoff, dirv = self.push_geom(state, mode)
        x, y, th, aj, fg = self.robot(state)
        want = np.arctan2(dirv[1], dirv[0]) + np.pi
        dth = wrap(want - th)
        if self.pstage == 0:
            tgt = standoff - 0.26 * dirv
            a, done = self.move_to(state, tgt[0], tgt[1], tol=0.012)
            a = np.array(a, dtype=np.float64)
            a[2] = np.clip(dth, -0.065, 0.065)
            a[3] = -0.1
            a[4] = -0.02
            if done and abs(dth) < 0.05 and aj < 0.245:
                self.pstage = 1
            return self.act(*a)
        tgt = standoff + 0.10 * dirv
        a, done = self.move_to(state, tgt[0], tgt[1], tol=0.004)
        if done or self.stall >= 3:
            self.pstage = 0
        return a

    # ----- grasping -----
    def ph_grasp_prep(self, state):
        """Open gripper, retract arm, align angle with hook (only when clear)."""
        x, y, th, aj, fg = self.robot(state)
        o, u, n, hth = self.hook_frame(state)
        dth = wrap(hth - th)
        if (abs(dth) > 0.01 and self.hook_dist(state, (x, y)) < 0.62
                and y > self.YMIN + 0.004 and self.stall < 3):
            return self.act(dy=-0.05, da=-0.1, dg=0.02)
        if abs(dth) < 0.01 and aj < 0.245 and fg > 0.315:
            self.phase = 'grasp_travel'
            return self.ph_grasp_travel(state)
        return self.act(dth=dth, da=-0.1, dg=0.02)

    def ph_grasp_travel(self, state):
        """Move to a staging point behind the grasp pose (safe lane), then in."""
        o, u, n, hth = self.hook_frame(state)
        base = self.grasp_base(state, self.d_grasp)
        stage = base - self.stage_gap(self.d_grasp) * u
        x, y = self.g(state, 'robot', 'x'), self.g(state, 'robot', 'y')
        if self.substep == 0:
            # travel along a safe lane below the hook
            lane = min(max(self.YMIN, o[1] - 0.62), self.YMAX)
            a, done = self.move_to(state, stage[0], lane, tol=0.008)
            if done:
                self.substep = 1
            return a
        if self.substep == 1:
            a, done = self.move_to(state, stage[0], stage[1], tol=0.006)
            if done:
                self.substep = 2
            return a
        a, done = self.move_to(state, base[0], base[1], tol=0.0025, scale=0.6)
        if done:
            self.phase = 'grasp_extend'
        return a

    def ph_grasp_extend(self, state):
        x, y, th, aj, fg = self.robot(state)
        want = self.d_grasp - self.GRASP_OFF
        if abs(aj - want) < 0.004:
            self.phase = 'grasp_close'
            return self.ph_grasp_close(state)
        return self.act(da=np.clip(want - aj, -0.1, 0.1))

    def ph_grasp_close(self, state):
        if self.g(state, 'hook', 'held') > 0.5:
            self._record_rel(state)
            self.phase = 'lift_prep'
            return self.ph_lift_prep(state)
        x, y, th, aj, fg = self.robot(state)
        if fg < 0.125:
            # failed; reopen and retry with a slightly different standoff
            self.phase = 'grasp_retry'
            self.substep = 0
            return self.act(dg=0.02)
        return self.act(dg=-0.02)

    def ph_grasp_retry(self, state):
        x, y, th, aj, fg = self.robot(state)
        if fg < 0.315:
            return self.act(dg=0.02)
        self.d_grasp = max(0.31, self.d_grasp - 0.02)
        self.phase = 'grasp_prep'
        self.substep = 0
        return self.ph_grasp_prep(state)

    def _record_rel(self, state):
        x, y, th, aj, fg = self.robot(state)
        o, u, n, hth = self.hook_frame(state)
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        self.rel_p = R.dot(o - np.array([x, y]))
        self.rel_th = wrap(hth - th)

    def base_for_hook(self, hook_pos, hook_th):
        th = wrap(hook_th - self.rel_th)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return np.array(hook_pos) - R.dot(self.rel_p), th

    def base_for_corner(self, state, goal):
        """Base position that puts the hook origin (corner) at goal (theta fixed)."""
        o = np.array([self.g(state, 'hook', 'x'), self.g(state, 'hook', 'y')])
        b = np.array([self.g(state, 'robot', 'x'), self.g(state, 'robot', 'y')])
        return b + (np.asarray(goal) - o)

    # ----- lift / rotate to vertical -----
    def rel_now(self, state):
        x, y, th, aj, fg = self.robot(state)
        o = np.array([self.g(state, 'hook', 'x'), self.g(state, 'hook', 'y')])
        hth = self.g(state, 'hook', 'theta')
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        return R.dot(o - np.array([x, y])), wrap(hth - th)

    def base_for_hook_pose(self, state, goal_pos, goal_hth):
        p, dth = self.rel_now(state)
        th = goal_hth - dth
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return np.asarray(goal_pos) - R.dot(p), th

    def base_for_corner(self, state, goal):
        o = np.array([self.g(state, 'hook', 'x'), self.g(state, 'hook', 'y')])
        b = np.array([self.g(state, 'robot', 'x'), self.g(state, 'robot', 'y')])
        return b + (np.asarray(goal) - o)

    def plan_tilt(self, state):
        """Choose ascent tilt beta and ascent base-x."""
        l, r, b, t = self.block_box(state, 'target_block')
        corner_x = l - 0.16
        final_bx = float(np.clip(corner_x + 0.5 * self.bar_w, self.XMIN, self.XMAX))
        # try no tilt first: pure vertical ascent corridor left of the block
        limit = l - self.stub_reach() - 0.04 - 0.5 * self.bar_w
        if min(final_bx, limit) >= self.XMIN + 0.002:
            return 0.0, float(min(final_bx, limit)), final_bx
        bx = max(self.XMIN, min(final_bx, self.XMAX))
        for beta in np.arange(0.08, 0.55, 0.02):
            left = bx - (0.55 + self.bar_len) * np.sin(beta)
            right = max(bx - 0.55 * np.sin(beta),
                        left + self.stub_reach() * np.cos(beta))
            if right < l - 0.04:
                return float(beta), float(bx), final_bx
        return 0.0, float(bx), final_bx

    def ph_lift_prep(self, state):
        beta, ascx, final_bx = self.plan_tilt(state)
        self.beta = beta
        self.asc_x = ascx
        self.final_bx = final_bx
        _, want_th = self.base_for_hook_pose(state, (0, 0), np.pi / 2 + beta)
        th = self.g(state, 'robot', 'theta')
        target = want_th
        while target > th - 0.05:
            target -= TWO_PI
        self.rot_goal = target
        self.phase = 'rotate'
        return self.ph_rotate(state)

    def ph_rotate(self, state):
        x, y, th, aj, fg = self.robot(state)
        d = self.rot_goal - th
        a, done = self.move_to(state, self.XMIN, self.YMIN, tol=0.01)
        if abs(d) < 0.004 and aj > 0.4795:
            self.phase = 'position'
            self.substep = 0
            return self.ph_position(state)
        return self.act(dx=a[0], dy=a[1], dth=np.clip(d, -0.065, 0.065),
                        da=0.1 if aj < 0.4795 else 0.0)

    # ----- position hook above target block -----
    def block_box(self, state, name):
        ex, ey = self.rect_extent(state, name)
        x = self.g(state, name, 'x')
        y = self.g(state, name, 'y')
        return x - ex, x + ex, y - ey, y + ey

    def hook_goal(self, state):
        l, r, b, t = self.block_box(state, 'target_block')
        return np.array([l - 0.16, t + 0.19])

    def stub_reach(self):
        return self.stub_len - 0.05

    def ph_position(self, state):
        x, y, th, aj, fg = self.robot(state)
        if self.substep == 0:
            goal = self.hook_goal(state)
            self.goal_corner = goal
            beta, ascx, final_bx = self.plan_tilt(state)
            self.asc_x = ascx
            self.final_bx = final_bx
            self.substep = 1
        if self.substep == 1:      # move horizontally, low
            a, done = self.move_to(state, self.asc_x, y, tol=0.006)
            if done:
                self.substep = 2
            return a
        if self.substep == 2:      # ascend so corner reaches goal height
            base = self.base_for_corner(state, np.array([x, self.goal_corner[1]]))
            a, done = self.move_to(state, self.asc_x, base[1], tol=0.006)
            if done:
                self.substep = 3
            return a
        if self.substep == 3:      # untilt
            hth = self.g(state, 'hook', 'theta')
            d = wrap(np.pi / 2 - hth)
            if abs(d) > 0.004:
                return self.act(dth=np.clip(d, -0.065, 0.065))
            self.substep = 4
        base = self.base_for_corner(state, self.goal_corner)
        final_by = float(np.clip(base[1], self.YMIN, self.YMAX))
        a, done = self.move_to(state, float(np.clip(base[0], self.XMIN, self.XMAX)),
                               final_by, tol=0.005)
        if done:
            self.phase = 'descend'
            self.desc_cnt = 0
            self._desc_prev = None
        return a

    def ph_descend(self, state):
        goal = self.hook_goal(state)
        base = self.base_for_corner(state, goal)
        x, y, th, aj, fg = self.robot(state)
        dxerr = base[0] - x
        if abs(dxerr) > 0.03 and self.stall < 4:
            return self.act(dx=np.clip(dxerr, -0.05, 0.05))
        dx = float(np.clip(dxerr, -0.02, 0.02))
        self.desc_cnt += 1
        prev = getattr(self, '_desc_prev', None)
        moved = prev is None or (prev[0] - y) > 0.002 or (prev[1] - aj) > 0.002
        if moved:
            self._desc_stall = 0
        else:
            self._desc_stall = getattr(self, '_desc_stall', 0) + 1
        self._desc_prev = (y, aj)
        if self._desc_stall < 4 and y > self.YMIN + 0.003:
            return self.act(dx=dx, dy=-0.05)
        if aj > 0.2405:
            if self._desc_stall > 12:
                self.replans += 1
                self.phase = 'lift_back'
                self._desc_prev = None
                return self.act(dy=0.05, da=0.1)
            return self.act(dx=dx, da=-0.1)
        self.replans += 1
        self.phase = 'lift_back'
        self._desc_prev = None
        return self.act(dy=0.05, da=0.1)

    def ph_lift_back(self, state):
        """Re-extend arm and rise back to engagement height, then descend."""
        x, y, th, aj, fg = self.robot(state)
        goal = self.hook_goal(state)
        base = self.base_for_corner(state, goal)
        by = float(np.clip(base[1], self.YMIN, self.YMAX))
        if aj < 0.4795:
            return self.act(dy=0.05 if y < by else 0.0, da=0.1)
        a, done = self.move_to(state, x, by, tol=0.01)
        if not done:
            return a
        self.phase = 'descend'
        self.desc_cnt = 0
        self._desc_prev = None
        self._desc_stall = 0
        return self.ph_descend(state)
