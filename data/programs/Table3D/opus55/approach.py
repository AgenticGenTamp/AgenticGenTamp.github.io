import numpy as np
from scipy.optimize import minimize

# ---------------- Kinova Gen3 kinematics (classic DH) ----------------
_D = [-(0.1564 + 0.1284), -(0.0054 + 0.0064), -(0.2104 + 0.2104), -(0.0064 + 0.0064),
      -(0.2084 + 0.1059), 0.0, -(0.1059 + 0.0615)]
_ALPHA = [np.pi / 2] * 6 + [np.pi]
_OFF = [0, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
_CA = [np.cos(a) for a in _ALPHA]
_SA = [np.sin(a) for a in _ALPHA]
_LIM = np.array([[-1e9, 1e9], [-2.40, 2.40], [-1e9, 1e9], [-2.65, 2.65],
                 [-1e9, 1e9], [-2.22, 2.22], [-1e9, 1e9]])

ARM_OFFSET_X = 0.12      # arm mount offset from base center (base frame)
GRASP_DZ = 0.235         # cube_center_z(world) - flange_z(arm frame) at grasp


def _dh(i, th):
    ca, sa, ct, st = _CA[i], _SA[i], np.cos(th), np.sin(th)
    return np.array([[ct, -st * ca, st * sa, 0.0],
                     [st, ct * ca, -ct * sa, 0.0],
                     [0.0, sa, ca, _D[i]],
                     [0.0, 0.0, 0.0, 1.0]])


def fk(q):
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    for i in range(7):
        T = T @ _dh(i, q[i] + _OFF[i])
    return T


def _rot_err(R, Rd):
    Re = Rd @ R.T
    return 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


def ik(q0, p_des, R_des, iters=150, wrot=0.5, tol=1e-5):
    q = np.array(q0, float)
    err = 1e9
    for _ in range(iters):
        T = fk(q)
        e = np.concatenate([p_des - T[:3, 3], wrot * _rot_err(T[:3, :3], R_des)])
        err = np.linalg.norm(e)
        if err < tol:
            break
        J = np.zeros((6, 7))
        eps = 1e-6
        for j in range(7):
            dq = q.copy()
            dq[j] += eps
            Tj = fk(dq)
            J[:3, j] = (Tj[:3, 3] - T[:3, 3]) / eps
            J[3:, j] = wrot * _rot_err(T[:3, :3], Tj[:3, :3]) / eps
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        q = np.clip(q + dq, _LIM[:, 0], _LIM[:, 1])
    return q, err


def down_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _quat_yaw(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))



# ---------------- fast (min-step) planner using base + arm ----------------
BX_MARGIN = 0.4 - 0.19   # base center x must satisfy bx + ext(rot) <= 0.4


def _base_ext(br):
    return 0.17 * abs(np.cos(br)) + 0.06 * abs(np.sin(br))


def world_flange(x):
    bx, by, br = x[0], x[1], x[2]
    T = fk(x[3:10])
    c, s = np.cos(br), np.sin(br)
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    p = Rb @ (T[:3, 3] + np.array([ARM_OFFSET_X, 0, 0])) + np.array([bx, by, GRASP_DZ])
    return p, Rb @ T[:3, :3]


def solve_min_step(x0, p_cube, yaw, table_x0=0.4, slack=None, bmargin=0.01):
    Rd = down_R(yaw)

    sl = None if slack is None else np.asarray(slack, float)

    def cons_eq(v):
        p, R = world_flange(v[:10])
        ori = np.concatenate([(R[:, 2] - Rd[:, 2])[:2], [R[:, 0] @ Rd[:, 1]]])
        if sl is None:
            return np.concatenate([p - p_cube, ori])
        return ori

    def cons_pos(v):
        if sl is None:
            return np.zeros(0)
        p, R = world_flange(v[:10])
        e = Rd.T @ (p - p_cube)
        return np.concatenate([sl - e, sl + e])

    def cons_in(v):
        d = v[:10] - x0
        s = v[10]
        return np.concatenate([s - d, s + d, [table_x0 - bmargin - v[0] - _base_ext(v[2])], cons_pos(v),
                               v[3:10] - _LIM[:, 0], _LIM[:, 1] - v[3:10]])
    # warm start: arm-only IK from current base pose
    bx, by, br = x0[0], x0[1], x0[2]
    c, s_ = np.cos(br), np.sin(br)
    dxw, dyw = p_cube[0] - bx, p_cube[1] - by
    pa = np.array([c * dxw + s_ * dyw - ARM_OFFSET_X, -s_ * dxw + c * dyw, p_cube[2] - GRASP_DZ])
    Ra = down_R(yaw - br)
    qi, ei = ik(x0[3:10], pa, Ra)
    # unwrap continuous joints toward current values
    for j in (0, 2, 4, 6):
        qi[j] = x0[3 + j] + ((qi[j] - x0[3 + j] + np.pi) % (2 * np.pi) - np.pi)
    xi = np.concatenate([x0[:3], qi]) if ei < 1e-3 else x0.copy()
    v0 = np.concatenate([xi, [np.abs(xi - x0).max() + 1e-3]])
    try:
        res = minimize(lambda v: v[10] + 1e-3 * np.sum((v[:10] - x0) ** 2), v0, method='SLSQP',
                       constraints=[{'type': 'eq', 'fun': cons_eq}, {'type': 'ineq', 'fun': cons_in}],
                       options={'maxiter': 150, 'ftol': 1e-10})
        v = res.x
    except Exception:
        return None
    if not np.all(np.isfinite(v)):
        return None
    ce = np.abs(cons_eq(v)).max()
    ci = cons_in(v).min()
    if ce > 1e-4 or ci < -1e-4:
        return None
    s = np.abs(v[:10] - x0).max()
    if s > 3.0:
        return None
    return v[:10], s


def best_lift(qg, max_d=0.395):
    """Joint delta (|d|<=max_d) maximizing flange height while staying roughly pointing down."""
    Tg = fk(qg)
    z0 = Tg[2, 3]
    ql, e = ik(qg, Tg[:3, 3] + np.array([0, 0, 0.15]), Tg[:3, :3])
    d0 = ql - qg
    m = np.abs(d0).max()
    if m <= max_d:
        return d0
    d0 = d0 * (max_d / m)

    def f(d):
        T = fk(qg + d)
        tilt = 1.0 + T[2, 2]  # 0 when pointing straight down
        dxy = T[:2, 3] - Tg[:2, 3]
        return -(T[2, 3] - z0) + 0.3 * tilt + 0.5 * np.sum(dxy ** 2)
    lo = np.maximum(-max_d, _LIM[:, 0] - qg)
    hi = np.minimum(max_d, _LIM[:, 1] - qg)
    try:
        res = minimize(f, np.clip(d0, lo, hi), method='L-BFGS-B', bounds=list(zip(lo, hi)),
                       options={'maxiter': 100})
        d = res.x
        if f(d) < f(d0):
            return d
    except Exception:
        pass
    return d0


# ---------------- relaxed-orientation planner ----------------
MOUNT_H = 0.3947
DEPTH = 0.162
# collision probe points in flange frame (fingertips + gripper body corners)
PTS = np.array([[sx*0.05, sy*0.012, 0.15] for sx in (-1,1) for sy in (-1,1)] +
               [[sx*0.045, sy*0.035, 0.09] for sx in (-1,1) for sy in (-1,1)])

def world_T(x):
    bx, by, br = x[0], x[1], x[2]
    T = fk(x[3:10])
    c, s = np.cos(br), np.sin(br)
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    p = Rb @ (T[:3, 3] + np.array([ARM_OFFSET_X, 0, 0])) + np.array([bx, by, MOUNT_H])
    return p, Rb @ T[:3, :3]

def solve_relaxed(x0, p_cube, yaw, tilt_max=0.5, slack=(0.004, 0.01, 0.014), table_z=0.4, bmargin=0.01, xi=None):
    sl = np.asarray(slack)
    ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    perp = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    ct = np.cos(tilt_max)
    def cin(v):
        p, R = world_T(v[:10])
        e = R.T @ (p_cube - p) - np.array([0, 0, DEPTH])
        pts = p + PTS @ R.T
        d = v[:10] - x0; s = v[10]
        return np.concatenate([s - d, s + d, sl - e, sl + e, [-R[2, 2] - ct],
                               pts[:, 2] - (table_z + 0.004),
                               [0.4 - bmargin - v[0] - _base_ext(v[2])],
                               v[3:10] - _LIM[:, 0], _LIM[:, 1] - v[3:10]])
    def ceq(v):
        p, R = world_T(v[:10])
        return np.array([R[:, 0] @ perp])
    if xi is None:
        # warm start from down-orientation IK
        bx, by, br = x0[:3]
        c, s_ = np.cos(br), np.sin(br)
        dxw, dyw = p_cube[0] - bx, p_cube[1] - by
        pa = np.array([c*dxw + s_*dyw - ARM_OFFSET_X, -s_*dxw + c*dyw, p_cube[2] - MOUNT_H + DEPTH])
        qi, ei = ik(x0[3:10], pa, down_R(yaw - br))
        for j in (0, 2, 4, 6):
            qi[j] = x0[3+j] + ((qi[j] - x0[3+j] + np.pi) % (2*np.pi) - np.pi)
        xi = np.concatenate([x0[:3], qi]) if ei < 1e-3 else x0.copy()
    v0 = np.concatenate([xi, [np.abs(xi - x0).max() + 1e-3]])
    try:
        res = minimize(lambda v: v[10] + 1e-3*np.sum((v[:10]-x0)**2), v0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': cin}, {'type': 'eq', 'fun': ceq}],
                       options={'maxiter': 200, 'ftol': 1e-10})
        v = res.x
    except Exception:
        return None
    if not np.all(np.isfinite(v)) or cin(v).min() < -1e-5 or abs(ceq(v)[0]) > 1e-4:
        return None
    s = np.abs(v[:10] - x0).max()
    if s > 3: return None
    return v[:10], s


JOINTS = [f'joint_{i}' for i in range(1, 8)]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ---------- helpers ----------
    def _robot(self, state):
        return state.get_object_from_name('robot')

    def _q(self, state):
        r = self._robot(state)
        return np.array([state.get(r, n) for n in JOINTS])

    def _base(self, state):
        r = self._robot(state)
        return (state.get(r, 'pos_base_x'), state.get(r, 'pos_base_y'), state.get(r, 'pos_base_rot'))

    def _cubes(self, state):
        out = []
        for name in sorted(state.get_object_names()):
            if not name.startswith('cube'):
                continue
            o = state.get_object_from_name(name)
            out.append((name, o))
        return out

    def _world_to_arm(self, state, p):
        bx, by, br = self._base(state)
        c, s = np.cos(br), np.sin(br)
        dx, dy = p[0] - bx, p[1] - by
        lx = c * dx + s * dy - ARM_OFFSET_X
        ly = -s * dx + c * dy
        return np.array([lx, ly, p[2] - GRASP_DZ]), br

    def _cube_info(self, state, o):
        p = np.array([state.get(o, 'pose_x'), state.get(o, 'pose_y'), state.get(o, 'pose_z')])
        yaw = _quat_yaw(state.get(o, 'pose_qx'), state.get(o, 'pose_qy'),
                        state.get(o, 'pose_qz'), state.get(o, 'pose_qw'))
        return p, yaw

    def _plan_for(self, state, name):
        o = state.get_object_from_name(name)
        p, cyaw = self._cube_info(state, o)
        pa, br = self._world_to_arm(state, p)
        q0 = self._q(state)
        best = None
        for k in range(4):
            yaw = cyaw - br + k * np.pi / 2
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            R = down_R(yaw)
            qg, eg = ik(q0, pa, R)
            if eg > 1e-3:
                continue
            qp, ep = ik(qg, pa + np.array([0, 0, 0.10]), R)
            if ep > 1e-3:
                continue
            cost = np.abs(qp - q0).sum() + abs(yaw - np.pi / 2) * 0.1
            if best is None or cost < best[0]:
                best = (cost, yaw, R, pa)
        return best

    def _x(self, state):
        return np.array(list(self._base(state)) + list(self._q(state)))

    def _tip_ok(self, x, p_cube_list):
        # gripper probe points must stay above the table top
        p, R = world_T(x)
        pts = p + PTS @ R.T
        return pts[:, 2].min() >= 0.403

    def _finger_clear(self, state, name, p, yaw, cubes):
        """Min clearance between finger footprints and other cubes (negative = overlap)."""
        ax = np.array([np.cos(yaw), np.sin(yaw)])
        clear = 1.0
        for m, o in cubes:
            if m == name:
                continue
            q, qyaw = self._cube_info(state, o)
            hx = state.get(o, 'half_extent_x')
            hy = state.get(o, 'half_extent_y')
            c, s = np.cos(qyaw), np.sin(qyaw)
            for sgn in (1.0, -1.0):
                f = p[:2] + sgn * 0.045 * ax - q[:2]
                lx, ly = c * f[0] + s * f[1], -s * f[0] + c * f[1]
                dx = max(abs(lx) - hx, 0.0)
                dy = max(abs(ly) - hy, 0.0)
                clear = min(clear, np.hypot(dx, dy) - 0.015)
        return clear

    def _waypoints(self, x0, xg, N, lim=0.399):
        """Waypoints x_1..x_N (x_N = xg) with per-step |dx|<=lim keeping fingertips above table."""
        wps = []
        prev = x0
        for i in range(1, N):
            rem = N - i
            xl = prev + (xg - prev) / (rem + 1)
            if self._tip_ok(xl, None):
                wps.append(xl)
                prev = xl
                continue
            lo = np.maximum(prev - lim, xg - lim * rem)
            hi = np.minimum(prev + lim, xg + lim * rem)
            if np.any(lo > hi):
                return None

            def f(x):
                p, R = world_flange(x)
                return -p[2]
            try:
                res = minimize(f, np.clip(xl, lo, hi), method='L-BFGS-B', bounds=list(zip(lo, hi)),
                               options={'maxiter': 60})
                xm = res.x
            except Exception:
                return None
            if not self._tip_ok(xm, None):
                return None
            wps.append(xm)
            prev = xm
        if np.abs(xg - prev).max() > lim + 1e-6:
            return None
        wps.append(xg)
        return wps

    def _fast_plan(self, state, bad=(), budget=20.0):
        import time as _time
        t0 = _time.time()
        x0 = self._x(state)
        cubes = self._cubes(state)
        cands = []
        for n, o in cubes:
            p, cyaw = self._cube_info(state, o)
            pa, _ = self._world_to_arm(state, p)
            reach = abs(np.linalg.norm(pa[:2]) - 0.35)
            for k in range(4):
                if (n, k) in bad:
                    continue
                yaw = cyaw + k * np.pi / 4
                cl = min(self._finger_clear(state, n, p, yaw, cubes), 0.01)
                cands.append((-cl, reach, k, n, p, yaw))
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        best = None
        for negcl, reach, k, n, p, yaw in cands:
            if best is not None:
                if negcl > best[0][0]:
                    break
                if best[0][1] <= 2 or _time.time() - t0 > budget:
                    break
            for solver in (0, 1):
                if solver == 0:
                    sol = solve_relaxed(x0, p, yaw)
                else:
                    if k % 2:
                        continue
                    sol = solve_min_step(x0, p, yaw, slack=(0.003, 0.012, 0.005))
                if sol is None:
                    continue
                xg, s = sol
                N = max(1, int(np.ceil(s / 0.4 - 1e-6)))
                key = (negcl, N, s)
                if best is not None and key >= best[0]:
                    continue
                wps = self._waypoints(x0, xg, N)
                if wps is None:
                    N += 1
                    key = (negcl, N, s)
                    if best is not None and key >= best[0]:
                        continue
                    wps = self._waypoints(x0, xg, N)
                    if wps is None:
                        continue
                best = (key, n, xg, N, k, wps)
                break
        if best is None:
            return None
        _, n, xg, N, kk, wps = best
        acts = []
        exp = []
        prev = x0
        for i in range(N):
            a = np.zeros(11, dtype=np.float32)
            a[:10] = np.clip(wps[i] - prev, -0.4, 0.4)
            if i == N - 1:
                a[10] = -1.0
            acts.append(a)
            prev = prev + a[:10].astype(np.float64)
            exp.append(prev.copy())
        # lift: joint-only IK raising the flange
        qg = xg[3:10]
        d = best_lift(qg)
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = d
        acts.append(a)
        exp.append(None)
        return (n, kk), acts, exp

    def reset(self, state, info):
        self.fast = None
        self.fast_i = 0
        self.bad = set()
        self.replans = 0
        try:
            self.fast = self._fast_plan(state)
        except Exception:
            self.fast = None
        self.phase = 'select'
        self.tried = set()
        self.target = None
        self.plan = None
        self.close_count = 0
        self.stuck = 0
        self.last_q = None
        self.wp = []

    def _select(self, state):
        cubes = self._cubes(state)
        pos = {n: self._cube_info(state, o)[0] for n, o in cubes}
        cands = []
        for n, o in cubes:
            if n in self.tried:
                continue
            p = pos[n]
            # clearance to other cubes
            dmin = min([np.linalg.norm((p - pos[m])[:2]) for m in pos if m != n] + [1.0])
            pa, _ = self._world_to_arm(state, p)
            reach = np.linalg.norm(pa[:2])
            score = -min(dmin, 0.12) * 10 + abs(reach - 0.45)
            cands.append((score, n))
        cands.sort()
        for _, n in cands:
            plan = self._plan_for(state, n)
            self.tried.add(n)
            if plan is not None:
                self.target = n
                self.plan = plan
                _, yaw, R, pa = plan
                # waypoints (arm-frame flange positions)
                self.wp = [('move', pa + np.array([0, 0, 0.12]), R),
                           ('move', pa + np.array([0, 0, 0.05]), R),
                           ('move', pa, R),
                           ('close', None, None),
                           ('lift', pa + np.array([0, 0, 0.25]), R)]
                self.phase = 'exec'
                return True
        return False

    def _act(self, dq=None, grip=0.0, dbase=None):
        a = np.zeros(11, dtype=np.float32)
        if dq is not None:
            a[3:10] = np.clip(dq, -0.4, 0.4)
        if dbase is not None:
            a[0:3] = np.clip(dbase, -0.4, 0.4)
        a[10] = grip
        return a

    def get_action(self, state):
        if self.fast is not None:
            name, acts, exp = self.fast
            i = self.fast_i
            ok = True
            if i > 0 and exp[i - 1] is not None:
                if np.abs(self._x(state) - exp[i - 1]).max() > 1e-3:
                    ok = False
                elif i - 1 == len(acts) - 2:
                    r = self._robot(state)
                    ok = state.get(r, 'grasp_active') > 0.5
            if ok and i < len(acts):
                self.fast_i += 1
                return acts[i].copy()
            self.fast = None
            r = self._robot(state)
            if self.replans < 3 and state.get(r, 'grasp_active') < 0.5:
                # replan from current state, excluding the failed candidate
                self.bad.add(name)
                self.replans += 1
                try:
                    self.fast = self._fast_plan(state, self.bad)
                except Exception:
                    self.fast = None
                self.fast_i = 0
                if self.fast is not None:
                    return self.get_action(state)
        q = self._q(state)
        r = self._robot(state)
        grasped = state.get(r, 'grasp_active') > 0.5
        if self.phase == 'select':
            if not self._select(state):
                self.tried = set()
                return self._act()
        if not self.wp:
            return self._act()
        kind, target, R = self.wp[0]
        if kind == 'close':
            if grasped:
                self.wp.pop(0)
                kind, target, R = self.wp[0]
            else:
                self.close_count += 1
                if self.close_count > 3:
                    # grasp failed; try another cube
                    self.close_count = 0
                    self.phase = 'select'
                    self.wp = []
                    return self._act(grip=1.0)
                return self._act(grip=-1.0)
        # Cartesian step: interpolate towards target
        T = fk(q)
        cur = T[:3, 3]
        d = target - cur
        n = np.linalg.norm(d)
        step = 0.06
        sub = target if n < step else cur + d * (step / n)
        qt, e = ik(q, sub, R)
        dq = qt - q
        if n < 2e-3 and np.abs(dq).max() < 1e-3:
            if kind != 'lift':
                self.wp.pop(0)
            return self.get_action(state) if self.wp and self.wp[0][0] != 'close' else self._act(dq)
        # stuck detection
        if self.last_q is not None and np.abs(q - self.last_q).max() < 1e-7:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_q = q.copy()
        if self.stuck > 3 and not grasped:
            self.stuck = 0
            self.phase = 'select'
            self.wp = []
        return self._act(dq, grip=0.0)
