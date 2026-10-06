

from scipy.optimize import linprog as _linprog
from scipy.sparse import csc_matrix as _csc
from scipy.optimize._highs._highs_wrapper import _highs_wrapper
from scipy.optimize._highs._highs_constants import (
    CONST_INF as _INF, MODEL_STATUS_OPTIMAL as _OPT, MESSAGE_LEVEL_NONE as _MLN,
    HIGHS_OBJECTIVE_SENSE_MINIMIZE as _MIN)
_LP_OPTS = {'presolve': True, 'sense': _MIN, 'solver': 'simplex', 'highs_debug_level': _MLN,
            'log_to_console': False, 'output_flag': False}


def _fast_lp(c, M, rhs, lb, ub):
    A = _csc(M)
    lhs = np.full(len(rhs), -_INF)
    r = _highs_wrapper(c, A.indptr, A.indices, A.data, lhs, rhs, lb, ub, np.empty(0, np.uint8), dict(_LP_OPTS))
    if r.get('status') != _OPT:
        return None
    return np.asarray(r['x'])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def base_region(q):
    """Half-plane keeping the base on the side of the table it is currently on."""
    x, y = q[0], q[1]
    if abs(x) / 0.46 >= abs(y) / 0.94:
        return ('x', np.sign(x) if x != 0 else -1.0)
    return ('y', np.sign(y))


class TaskGrasp:
    """Block centre c inside the gripper; tool x-axis ~down; yaw near target."""

    def __init__(self, c, yaw, yaw_tol=0.35, tilt_tol=0.25, ox=(0.012, 0.042), oy=0.003, oz=0.02):
        self.c = np.asarray(c, float)
        self.Rt = np.array(p.getMatrixFromQuaternion(top_quat(yaw))).reshape(3, 3)
        self.yaw_tol, self.tilt_tol = yaw_tol, tilt_tol
        self.ox, self.oy, self.oz = ox, oy, oz

    def rows(self, F, J):
        R, pt = F[:3, :3], F[:3, 3]
        v = self.c - pt
        o = R.T @ v
        Ao = R.T @ (-J[:3] + skew(v) @ J[3:])
        r = rot_err(self.Rt, R)
        Ar = -J[3:]
        b = np.concatenate([o, r])
        A = np.vstack([Ao, Ar])
        lo = np.array([self.ox[0], -self.oy, -self.oz, -self.tilt_tol, -self.tilt_tol, -self.yaw_tol])
        hi = np.array([self.ox[1], self.oy, self.oz, self.tilt_tol, self.tilt_tol, self.yaw_tol])
        return A, b, lo, hi


class TaskPlace:
    """Held block (grasp G) centred over (sx,sy) within tol, upright, yaw near target."""

    def __init__(self, G, sx, sy, yaw, tol_xy, yaw_tol, z=(0.79, 0.86), tilt_tol=0.01):
        self.gp = np.asarray(G[0], float)
        self.Rg = np.array(p.getMatrixFromQuaternion(G[1])).reshape(3, 3)
        self.Rt = np.array(p.getMatrixFromQuaternion(yaw_quat(yaw))).reshape(3, 3)
        self.s = np.array([sx, sy])
        self.tol, self.yaw_tol, self.z, self.tilt_tol = tol_xy, yaw_tol, z, tilt_tol

    def rows(self, F, J):
        R, pt = F[:3, :3], F[:3, 3]
        rg = R @ self.gp
        bpos = pt + rg
        Ab = J[:3] - skew(rg) @ J[3:]
        Rb = R @ self.Rg
        r = rot_err(self.Rt, Rb)
        Ar = -J[3:]
        b = np.concatenate([bpos, r])
        A = np.vstack([Ab, Ar])
        t = self.tol
        lo = np.array([self.s[0] - t[0], self.s[1] - t[1], self.z[0], -self.tilt_tol, -self.tilt_tol, -self.yaw_tol])
        hi = np.array([self.s[0] + t[0], self.s[1] + t[1], self.z[1], self.tilt_tol, self.tilt_tol, self.yaw_tol])
        return A, b, lo, hi


def slp_solve(sim, qc, q0, task, region=None, iters=12, shrink=0.8, margin=0.85, stop_steps=1e9):
    """Minimise max_i |q_i - qc_i| subject to task box constraints (linearised iteratively)."""
    ff = sim.ff
    q = np.array(q0, float)
    lo_j, hi_j = sim.lo, sim.hi
    rho = 0.4
    best = None
    for it in range(iters):
        F, J = ff.jac(q)
        A, b, lo, hi = task.rows(F, J)
        mid = (lo + hi) / 2
        half = (hi - lo) / 2
        lo_m, hi_m = mid - half * margin, mid + half * margin
        viol = np.maximum(0, np.maximum(lo - b, b - hi))
        feas = np.all(viol < 1e-9)
        delta = conf_delta(qc, q)
        tval = np.max(np.abs(delta))
        if feas and (best is None or tval < best[1] - 1e-6):
            best = (q.copy(), tval)
        m = A.shape[0]
        nv = 10 + 1 + m
        c = np.zeros(nv)
        c[10] = 1.0
        c[11:] = 100.0
        nr = 20 + 2 * m + (1 if region is not None else 0)
        M = np.zeros((nr, nv))
        I = np.eye(10)
        M[0:10, :10] = I; M[0:10, 10] = -1
        M[10:20, :10] = -I; M[10:20, 10] = -1
        rhs = np.empty(nr)
        rhs[0:10] = -delta
        rhs[10:20] = delta
        Em = np.eye(m)
        M[20:20 + m, :10] = A; M[20:20 + m, 11:] = -Em
        M[20 + m:20 + 2 * m, :10] = -A; M[20 + m:20 + 2 * m, 11:] = -Em
        rhs[20:20 + m] = hi_m - b
        rhs[20 + m:20 + 2 * m] = b - lo_m
        lbv = np.zeros(nv); ubv = np.full(nv, _INF)
        l = np.full(10, -rho); h = np.full(10, rho)
        fin = np.isfinite(lo_j)
        l[fin] = np.maximum(l[fin], (lo_j - q)[fin])
        fin = np.isfinite(hi_j)
        h[fin] = np.minimum(h[fin], (hi_j - q)[fin])
        lbv[:10] = np.minimum(l, 0.0); ubv[:10] = np.maximum(h, 0.0)
        if region is not None:
            ax, sg = region[0], region[1]
            i = 0 if ax == 'x' else 1
            lim = region[2] if len(region) > 2 else (0.47 if ax == 'x' else 0.95)
            M[nr - 1, i] = -sg
            rhs[nr - 1] = sg * q[i] - lim
        res = _fast_lp(c, M, rhs, lbv, ubv)
        if res is None:
            break
        t_lp = res[10]
        lb = int(np.ceil(t_lp / 0.2 - 1e-6))
        slack = np.sum(res[11:])
        if slack < 1e-6:
            if lb >= stop_steps and it >= 1:
                break
            if best is not None and int(np.ceil(best[1] / 0.2 - 1e-6)) <= lb:
                break
        dq = res[:10]
        q = q + dq
        q = np.clip(q, lo_j, hi_j)
        for d in (2, 7, 9):
            q[d] = wrap(q[d])
        rho *= shrink
    F, J = ff.jac(q)
    A, b, lo, hi = task.rows(F, J)
    if np.all(b >= lo - 1e-9) and np.all(b <= hi + 1e-9):
        tval = np.max(np.abs(conf_delta(qc, q)))
        if best is None or tval < best[1]:
            best = (q.copy(), tval)
    return best


def slp_safe(sim, planner, qc, q0, task, region, tries=4, stop_steps=1e9):
    ax, sg = region[0], region[1]
    i = 0 if ax == 'x' else 1
    lim = 0.47 if ax == 'x' else 0.95
    for _ in range(tries):
        r = slp_solve(sim, qc, q0, task, (ax, sg, lim), stop_steps=stop_steps)
        if r is None:
            return None
        if not planner.table_collision(r[0]):
            return r
        lim = max(lim, sg * r[0][i]) + 0.05
        q0 = r[0]
    return None
