import numpy as np

# ---------------- Kinova Gen3 kinematics (classic DH) ----------------
_DH = [(np.pi/2, 0, -0.2848, 0.0), (np.pi/2, 0, -0.0118, np.pi), (np.pi/2, 0, -0.4208, np.pi),
       (np.pi/2, 0, -0.0128, np.pi), (np.pi/2, 0, -0.3143, np.pi), (np.pi/2, 0, 0.0, np.pi),
       (np.pi, 0, -0.1674, np.pi)]
_T0 = np.diag([1., -1., -1., 1.])


def _T(alpha, a, d, th):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(th), np.sin(th)
    return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])


def fk_arm(q):
    T = _T0.copy()
    for (al, a, d, off), qi in zip(_DH, q):
        T = T @ _T(al, a, d, qi + off)
    return T


def _jac(q, eps=1e-6):
    T = fk_arm(q)
    J = np.zeros((6, 7))
    for i in range(7):
        dq = np.array(q, float); dq[i] += eps
        T2 = fk_arm(dq)
        J[:3, i] = (T2[:3, 3] - T[:3, 3]) / eps
        dR = T2[:3, :3] @ T[:3, :3].T
        J[3:, i] = np.array([dR[2, 1]-dR[1, 2], dR[0, 2]-dR[2, 0], dR[1, 0]-dR[0, 1]]) / (2*eps)
    return T, J


def _rot_err(R, Rd):
    E = Rd @ R.T
    return 0.5*np.array([E[2, 1]-E[1, 2], E[0, 2]-E[2, 0], E[1, 0]-E[0, 1]])


def ik(p_des, R_des, q0, iters=150, rot_w=0.3):
    q = np.array(q0, float)
    ep = np.zeros(3)
    for _ in range(iters):
        T, J = _jac(q)
        ep = p_des - T[:3, 3]
        er = _rot_err(T[:3, :3], R_des)
        if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
            break
        e = np.concatenate([ep, rot_w*er])
        Jw = J.copy(); Jw[3:] *= rot_w
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-3*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3/n
        q += dq
    return q, float(np.linalg.norm(ep))


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


MX = 0.1193    # arm mount x offset (base frame)
H = 0.395      # arm mount height
L = 0.145      # flange -> grasped cube center along tool z
LIM = np.array([9, 2.2, 9, 2.53, 9, 2.05, 9])  # Gen3 limits (j2 2.25, j4 2.58, j6 2.10) minus margin
HOME = np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
RF = ['pos_base_x', 'pos_base_y', 'pos_base_rot'] + ['pos_arm_joint%d' % i for i in range(1, 8)] + ['pos_gripper']
SHELF2 = 0.546  # upper interior shelf surface height relative to cupboard z
Q_SEEDS = [np.array(q) for q in [
    [0, 1.3, 3.14, -1.54, 0, -0.3, 1.57],
    [-0.01, 1.7, 3.15, -1.95, -0.01, 2.08, 1.56],
    [-0.12, 0.95, 3.31, -2.45, -0.03, 1.82, 1.43],
    [0, 0.3, 3.14, -2.0, 0, 1.0, 1.57]]]


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def Rpitch(p, yaw=0.0):
    """Tool z points forward (+x of base) tilted down by p; tool x (finger closing axis) horizontal."""
    z = np.array([np.cos(p), 0, -np.sin(p)])
    x = np.array([0, 1., 0])
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    return Rz(yaw) @ R


def ik_limited(p_des, R_des, q0, iters=200, rot_w=0.5):
    q = np.array(q0, float)
    f = np.array(p_des) - np.array([MX, 0, H]) - L*R_des[:, 2]
    ep = er = np.ones(3)
    for _ in range(iters):
        T, J = _jac(q)
        ep = f - T[:3, 3]
        er = _rot_err(T[:3, :3], R_des)
        if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 2e-3:
            break
        e = np.concatenate([ep, rot_w*er])
        Jw = J.copy(); Jw[3:] *= rot_w
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-3*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.2:
            dq *= 0.2/n
        q = np.clip(q + dq, -LIM, LIM)
    return q, float(np.linalg.norm(ep) + 0.2*np.linalg.norm(er))


def solve(p_des, R_des, seeds):
    best = None
    for s in seeds:
        q, e = ik_limited(p_des, R_des, s)
        if best is None or e < best[1]:
            best = (q, e)
        if e < 2e-3:
            break
    return best


def cube_pos_from_q(q):
    T = fk_arm(q)
    return T[:3, 3] + np.array([MX, 0, H]) + L*T[:3, 2]


_CONT = np.array([True, False, True, False, True, False, True])


def qerr(qt, q):
    """joint error; wrap only continuous joints (1,3,5,7)"""
    e = np.array(qt) - np.array(q)
    return np.where(_CONT, wrap(e), e)


def world_to_base(p, b):
    c, s = np.cos(b[2]), np.sin(b[2]); d = np.array(p[:2]) - b[:2]
    return np.array([c*d[0]+s*d[1], -s*d[0]+c*d[1], p[2]])


def quat_yaw(qw, qx, qy, qz):
    return np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))


def slots_for(n, cup_xy):
    """Target (x, y) positions on the shelf for n cubes, placement order (back first)."""
    cx, cy = cup_xy
    if n <= 4:
        ys = np.linspace(-0.2, 0.2, n) if n > 1 else np.array([0.0])
        return [(cx, cy + y) for y in ys]
    per = int(np.ceil(n / 2))
    ys = np.linspace(-0.21, 0.21, per)
    out = [(cx + 0.07, cy + y) for y in ys]
    rest = n - per
    ys2 = np.linspace(-0.21, 0.21, rest) if rest > 1 else np.array([0.0])
    out += [(cx - 0.07, cy + y) for y in ys2]
    return out


# ---- tunables ----
PICK_D = 0.770
PICK_PITCH = np.radians(77.5)
PLACE_PITCH = np.radians(9.7)
PLACE_D = 1.15
PLACE_ZOFF = 0.06
# joint-space seeds from offline optimization (min max|dq| between pick and place, HOME branch)
Q_PICK0 = np.array([0.010, 1.645, 3.123, -0.992, 0.056, -0.286, 1.533])
Q_PLACE_LO = np.array([0.002, 1.469, 3.138, -0.358, 0.011, 0.086, 1.595])
Q_PLACE_HI = np.array([0.008, 1.345, 3.092, -0.328, 0.158, -0.069, 1.492])
PRE_BACK = 0.30      # base retreat distance before insert
HOVER = 0.03
APPROACH_TOL = 0.12
INS_EQ = 0.04
PRE_GAP = 0.24
ARM_GAIN = 2.0
SMALL_E = 0.04
SMALL_G = 2.0
GRIP_STEPS = 2
REL_STEPS = 2
FIRST_PICK = None   # optional (PICK_D, PICK_PITCH, q_hover_seed) used for the first pick only


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    def _parse(self, state):
        robot = None
        cubes = {}
        cup = None
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            tn = o.type.name if hasattr(o.type, 'name') else str(o.type)
            if 'robot' in tn:
                robot = np.array([state.get(o, f) for f in RF])
            elif 'movable' in tn:
                cubes[name] = np.array([state.get(o, f) for f in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']])
            elif 'fixture' in tn:
                cup = np.array([state.get(o, f) for f in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']])
        return robot, cubes, cup

    def reset(self, state, info):
        robot, cubes, cup = self._parse(state)
        self.cup = cup if cup is not None else np.array([1.5, 0, 0, 1, 0, 0, 0])
        self.shelf_z = self.cup[2] + SHELF2
        self.slots = slots_for(len(cubes), self.cup[:2])
        self.slot_i = 0
        self.placed = set()
        self._depth = 0
        self._qprev = None
        self.cur = None
        self.phase = 'select'
        self.cnt = 0
        self.base_t = robot[:3].copy()
        self.q_t = robot[3:10].copy()
        self.grip = 0.0
        self.inserting = False
        # precompute place config (cube held at (PLACE_D, 0, shelf+0.025) in base frame)
        zc = self.shelf_z + PLACE_ZOFF
        w = np.clip((zc - 0.59)/0.10, -0.5, 1.5)
        qs = None if Q_PLACE_LO is None else (1-w)*Q_PLACE_LO + w*Q_PLACE_HI
        self.q_place, _ = solve(np.array([PLACE_D, 0, zc]), Rpitch(PLACE_PITCH), Q_SEEDS if qs is None else ([qs] + Q_SEEDS))
        Rp = Rpitch(PICK_PITCH)
        self.q_hover, _ = solve(np.array([PICK_D, 0, 0.02]) - HOVER*Rp[:, 2], Rp, ([] if Q_PICK0 is None else [Q_PICK0]) + [self.q_place] + Q_SEEDS)
        self.q_carry = self.q_place
        self.first = FIRST_PICK is not None
        self.pick_d, self.pick_p, self.q_hov_cur = PICK_D, PICK_PITCH, self.q_hover
        if self.first:
            fd, fp, fq = FIRST_PICK
            Rf = Rpitch(fp)
            qf, ef = solve(np.array([fd, 0, 0.02]) - HOVER*Rf[:, 2], Rf, [np.array(fq)] + Q_SEEDS)
            self.pick_d, self.pick_p, self.q_hov_cur = fd, fp, qf

    # ------------------------------------------------------------------
    def _in_cup_zone(self, p, margin=0.0):
        return (p[0] > self.cup[0] - 0.16 - margin and p[0] < self.cup[0] + 0.25 + margin
                and abs(p[1] - self.cup[1]) < 0.42 + margin)

    def _tip_world(self, base, q):
        T = fk_arm(q)
        pb = T[:3, 3] + np.array([MX, 0, H]) + (L + 0.03)*T[:3, 2]
        c, s = np.cos(base[2]), np.sin(base[2])
        return np.array([base[0] + c*pb[0] - s*pb[1], base[1] + s*pb[0] + c*pb[1], pb[2]])

    def _act(self, robot):
        a = np.zeros(11, dtype=np.float32)
        eb = self.base_t - robot[:3]
        eb[2] = wrap(eb[2])
        eq = qerr(self.q_t, robot[3:10])
        a[:3] = np.clip(eb, -0.1, 0.1)
        g = np.where(np.abs(eq) < SMALL_E, SMALL_G, ARM_GAIN)
        a[3:10] = np.clip(g*eq, -0.1, 0.1)
        a[10] = self.grip
        if self.phase not in ('insert', 'release', 'retreat') and not self.inserting:
            # guard: don't sweep the gripper into the cupboard
            nb = robot[:3] + a[:3]
            tip = self._tip_world(nb, robot[3:10] + 0.3*a[3:10])
            if self._in_cup_zone(tip):
                cur = self._tip_world(robot[:3], robot[3:10])
                a[2] = 0.0
                if self._in_cup_zone(cur):
                    a[0] = -0.1
                    a[1] = 0.0
                else:
                    a[0] = min(a[0], 0.0)
                    nb = robot[:3] + a[:3]
                    if self._in_cup_zone(self._tip_world(nb, robot[3:10])):
                        a[:3] = 0.0
        return a

    def _reached(self, robot, btol=0.015, qtol=0.02):
        eb = self.base_t - robot[:3]
        eb[2] = wrap(eb[2])
        eq = qerr(self.q_t, robot[3:10])
        return np.max(np.abs(eb[:2])) < btol and abs(eb[2]) < 0.03 and np.max(np.abs(eq)) < qtol

    def _free_slot(self, cubes):
        occ = [c[:2] for n, c in cubes.items() if n != self.cur and
               self.shelf_z - 0.01 < c[2] < self.shelf_z + 0.12]
        best, bd = 0, -1.0
        for i, sl in enumerate(self.slots):
            d = min([np.linalg.norm(np.asarray(sl) - o) for o in occ], default=9.0)
            if d > 0.06:
                return i
            if d > bd:
                best, bd = i, d
        return best

    def _pick_base(self, robot, c):
        cy = quat_yaw(*c[3:7])
        best = None
        for k in range(4):
            th = cy + k*np.pi/2
            bxy = c[:2] - self.pick_d*np.array([np.cos(th), np.sin(th)])
            cost = np.linalg.norm(bxy - robot[:2]) + 0.3*abs(wrap(th - robot[2]))
            # base footprint must stay clear of the cupboard
            if bxy[0] > self.cup[0] - 0.2 - 0.45 and abs(bxy[1] - self.cup[1]) < 0.42 + 0.45:
                cost += 100
            if best is None or cost < best[0]:
                best = (cost, bxy, th)
        _, bxy, th = best
        return np.array([bxy[0], bxy[1], robot[2] + wrap(th - robot[2])])

    def _grasp_yaw(self, c, base_th):
        cy = quat_yaw(*c[3:7])
        return ((cy - base_th + np.pi/4) % (np.pi/2)) - np.pi/4

    def get_action(self, state):
        robot, cubes, cup = self._parse(state)
        self.cnt += 1
        ph = self.phase
        if ph == 'select':
            # drop cubes that are not actually resting on the target shelf
            self.placed = {n for n in self.placed if n in cubes and
                           self.shelf_z - 0.01 < cubes[n][2] < self.shelf_z + 0.12}
            remaining = [n for n in cubes if n not in self.placed]
            if not remaining:
                self.base_t = robot[:3].copy(); self.q_t = robot[3:10].copy()
                return self._act(robot)
            # nearest remaining cube
            self.cur = min(remaining, key=lambda n: np.linalg.norm(cubes[n][:2] - robot[:2]))
            c = cubes[self.cur]
            self.base_t = self._pick_base(robot, c)
            self.q_t = self.q_hov_cur.copy()
            self.grip = 0.0
            self.phase = 'approach'; self.cnt = 0
        c = cubes[self.cur]
        if self.phase == 'approach':
            # keep updating base target (cube could move)
            if self._reached(robot, btol=0.02, qtol=APPROACH_TOL) or self.cnt > 250:
                cb = world_to_base(c[:3], robot)
                yg = self._grasp_yaw(c, robot[2])
                self.q_t, _ = solve(np.array([cb[0], cb[1], c[2]]), Rz(yg) @ Rpitch(self.pick_p), [robot[3:10]] + Q_SEEDS)
                self.base_t = robot[:3].copy()
                self.phase = 'descend'; self.cnt = 0; self._qprev = None
        elif self.phase == 'descend':
            q = robot[3:10]
            # stalled or in a period-2 limit cycle around the target
            hist = self._qprev if self._qprev is not None else []
            stalled = (len(hist) >= 2 and min(np.max(np.abs(q - hist[-1])), np.max(np.abs(q - hist[-2]))) < 0.004
                       and np.max(np.abs(qerr(self.q_t, q))) < 0.05)
            self._qprev = (list(hist) + [q.copy()])[-2:]
            if self._reached(robot, qtol=0.02) or stalled or self.cnt > 40:
                self.grip = 1.0
                self.phase = 'grasp'; self.cnt = 0
        elif self.phase == 'grasp':
            if self.cnt >= GRIP_STEPS:
                if self.first:
                    self.first = False
                    self.pick_d, self.pick_p, self.q_hov_cur = PICK_D, PICK_PITCH, self.q_hover
                self.slot_i = self._free_slot(cubes)
                sx, sy = self.slots[self.slot_i]
                self.pre_base = np.array([self.cup[0] - PRE_GAP - PLACE_D, sy, 0.0])
                self.base_t = self.pre_base.copy()
                self.base_t[2] = robot[2] + wrap(0.0 - robot[2])
                self.q_t = self.q_place.copy()
                self.phase = 'transit'; self.cnt = 0
        elif self.phase == 'transit':
            sx, sy = self.slots[self.slot_i]
            if self.cnt >= 10 and c[2] < 0.035:
                # grasp failed or cube dropped -> retry
                self.grip = 0.0
                self.phase = 'select'
            else:
                eq = np.max(np.abs(qerr(self.q_t, robot[3:10])))
                ready = eq < INS_EQ and c[2] > self.shelf_z + 0.024 and abs(wrap(robot[2] - self.base_t[2])) < 0.03 \
                    and abs(robot[1] - sy) < 0.03
                ready = ready or self.inserting
                self.inserting = ready
                bx = sx - PLACE_D if ready else self.pre_base[0]
                self.base_t = np.array([bx, sy, self.base_t[2]])
                if ready and self._reached(robot, btol=0.01, qtol=0.06):
                    self.grip = 0.0
                    self.phase = 'release'; self.cnt = 0
                elif self.cnt > 300:
                    self.grip = 0.0
                    self.phase = 'release'; self.cnt = 0
        elif self.phase == 'release':
            self.inserting = False
            if self.cnt >= REL_STEPS:
                self.base_t = self.base_t.copy()
                self.base_t[0] = self.cup[0] - 0.2 - PLACE_D
                self.phase = 'retreat'; self.cnt = 0
        elif self.phase == 'retreat':
            if self._reached(robot, btol=0.05, qtol=1.0) or self.cnt > 40:
                self.placed.add(self.cur)
                self.phase = 'select'; self.cnt = 0
        if self.phase == 'select':
            if self._depth < 2 and any(n not in self.placed for n in cubes):
                self._depth += 1
                try:
                    return self.get_action(state)
                finally:
                    self._depth -= 1
            return self._act(robot)
        return self._act(robot)
