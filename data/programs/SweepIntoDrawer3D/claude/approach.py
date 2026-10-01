"""Policy for SweepIntoDrawer3D-o5: open an island drawer, sweep cubes into it."""
import numpy as np
from fk import fk, rotz
from ik import jacobian

RDOWN = np.array([[1.0,0,0],[0,-1,0],[0,0,-1]])
RFWD  = np.array([[0.0,0,1],[0,1,0],[-1,0,0]])
MOUNT = np.array([0.1205, 0.0, 0.2598])
CTOP  = 0.4598          # counter top height
EDGEX = 0.90            # near edge of the counter (world x), robot side
FACEX = 0.90            # drawer front face
HANDX = 0.945           # handle front
DRW   = slice(103,109)

def arm_from_world(base, p_w):
    return rotz(base[2]).T @ (np.asarray(p_w,float) - np.array([base[0],base[1],0.0])) - MOUNT

class Task(object):
    def __init__(self, kind, **kw):
        self.kind = kind; self.t = 0; self.__dict__.update(kw)

class GeneratedApproach(object):
    def __init__(self, action_space=None, observation_space=None, primitives=None):
        self.action_space = action_space

    # ------------------------------------------------------------------
    def reset(self, state, info=None):
        s = np.asarray(state, float)
        self.t = 0
        self.grip = 1.0
        self.cur = None
        self.drawer_y = None
        self.drw0 = s[DRW].copy()
        self.cand = 0
        self.queue = self.plan_open(s, 0)
        self.phase = 'open'
        self.pushed = 0

    # ---------------- plans ----------------
    def cand_list(self, s):
        cy = float(np.mean(s[[1,17,33,49,65]]))
        ys = sorted([-0.3, 0.3, -0.9], key=lambda y: abs(y-cy))
        out = []
        for z in (0.36, 0.26):
            for y in ys[:2]:
                out.append((y, z))
        return out

    def plan_open(self, s, i):
        cl = self.cand_list(s)
        if i >= len(cl):
            return []
        y, z = cl[i]
        bx = 1.72
        by = float(np.clip(y, -0.45, 0.45))
        yaw = float(s[127])
        q = []
        q.append(Task('base', target=[bx, by, yaw], grip=1.0, maxsteps=60,
                      qhold=np.array([0.,-0.35,3.14,-2.55,0.,-0.87,1.57])))
        q.append(Task('arm', p=[1.20, y, z+0.04], R=RFWD, grip=1.0, maxsteps=170, tol=0.02))
        q.append(Task('arm', p=[HANDX+0.005, y, z], R=RFWD, grip=1.0, maxsteps=90, tol=0.012))
        q.append(Task('grip', value=0.0, hold=8))
        q.append(Task('base', target=[bx+0.30, by, yaw], grip=0.0, maxsteps=45, qhold=None))
        q.append(Task('check', hold=2))
        return q

    def plan_sweep(self, s):
        """push cubes one at a time over the near edge above the open drawer"""
        yd = self.drawer_y if self.drawer_y is not None else -0.3
        cubes = s[:80].reshape(5,16)
        on = [c for c in cubes if c[2] > 0.40 and c[0] < EDGEX + 0.02]
        if not on:
            return []
        on.sort(key=lambda c: -c[0])
        c = on[0]
        cx, cy = float(c[0]), float(c[1])
        bx = float(np.clip(cx + 0.55, 1.20, 1.75))
        by = float(np.clip((cy + yd)*0.5, -0.5, 0.5))
        yaw = float(s[127])
        zs = CTOP + 0.028
        start = [cx - 0.10, cy, zs]
        midx = min(cx + 0.10, EDGEX - 0.05)
        q = []
        q.append(Task('base', target=[bx, by, yaw], grip=0.0, maxsteps=40, qhold=None))
        q.append(Task('arm', p=[start[0], start[1], 0.62], R=RDOWN, grip=0.0, maxsteps=120, tol=0.03))
        q.append(Task('arm', p=start, R=RDOWN, grip=0.0, maxsteps=70, tol=0.015))
        q.append(Task('arm', p=[midx, (cy+yd)*0.5, zs], R=RDOWN, grip=0.0, maxsteps=70, tol=0.02))
        q.append(Task('arm', p=[EDGEX + 0.06, yd, zs], R=RDOWN, grip=0.0, maxsteps=80, tol=0.03))
        q.append(Task('arm', p=[EDGEX + 0.06, yd, 0.60], R=RDOWN, grip=0.0, maxsteps=60, tol=0.05))
        return q

    # ---------------- execution ----------------
    def get_action(self, state):
        s = np.asarray(state, float)
        self.t += 1
        a = np.zeros(11, dtype=np.float64)
        for _ in range(6):
            if self.cur is None:
                if not self.queue:
                    self.next_phase(s)
                if not self.queue:
                    a[10] = self.grip; return self.fin(a)
                self.cur = self.queue.pop(0); self.cur.t = 0
            if self.finished(self.cur, s):
                self.cur = None; continue
            break
        task = self.cur; task.t += 1
        if task.kind == 'arm':
            a[3:10] = self.servo_cmd(s, task.p, task.R)
            if task.grip is not None: self.grip = task.grip
        elif task.kind == 'base':
            d = np.asarray(task.target, float) - s[125:128]
            d[2] = (d[2]+np.pi) % (2*np.pi) - np.pi
            a[0] = np.clip(d[0]/0.8704, -0.1, 0.1)
            a[1] = np.clip(d[1]/0.8704, -0.1, 0.1)
            a[2] = np.clip(d[2]/0.994, -0.1, 0.1)
            if getattr(task, 'qhold', None) is not None:
                a[3:10] = np.clip(task.qhold - s[128:135], -0.1, 0.1)
            if task.grip is not None: self.grip = task.grip
        elif task.kind == 'grip':
            self.grip = task.value
        a[10] = self.grip
        return self.fin(a)

    def fin(self, a):
        a = np.clip(a, -0.1, 0.1)
        a[10] = float(np.clip(self.grip, 0.0, 1.0))
        if self.action_space is not None:
            return np.clip(a, self.action_space.low, self.action_space.high).astype(self.action_space.dtype)
        return a.astype(np.float32)

    def servo_cmd(self, s, p_w, R):
        pa = arm_from_world(s[125:128], p_w)
        J, M = jacobian(s[128:135])
        ep = pa - M[:3,3]
        Rerr = R @ M[:3,:3].T
        ang = np.arccos(np.clip((np.trace(Rerr)-1)/2, -1, 1))
        ax = np.zeros(3)
        if ang > 1e-8:
            ax = np.array([Rerr[2,1]-Rerr[1,2], Rerr[0,2]-Rerr[2,0], Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))*ang
        dq = J.T @ np.linalg.solve(J@J.T + 0.0064*np.eye(6), np.concatenate([ep, 0.3*ax]))
        return np.clip(dq*1.2, -0.1, 0.1)

    def perr(self, s, p_w):
        pa = arm_from_world(s[125:128], p_w)
        return float(np.linalg.norm(pa - fk(s[128:135])[:3,3]))

    def finished(self, task, s):
        if task.t >= getattr(task, 'maxsteps', 200): return True
        if task.kind == 'grip': return task.t >= getattr(task, 'hold', 6)
        if task.kind == 'check': return task.t >= 1
        if task.kind == 'arm':
            e = self.perr(s, task.p)
            if e < getattr(task, 'tol', 0.02): return True
            h = getattr(task, 'hist', None)
            if h is None: h = task.hist = []
            h.append(e)
            if len(h) > 25 and h[-25]-e < 0.004: return True
            return False
        if task.kind == 'base':
            d = np.asarray(task.target, float) - s[125:128]
            d[2] = (d[2]+np.pi) % (2*np.pi) - np.pi
            return abs(d[0]) < 0.005 and abs(d[1]) < 0.005 and abs(d[2]) < 0.02
        return True

    def next_phase(self, s):
        if self.phase == 'open':
            moved = np.abs(s[DRW] - self.drw0)
            if moved.max() > 0.06:
                idx = int(np.argmax(moved))
                self.drawer_y = {0: 0.3, 3: 0.3, 1: -0.3, 4: -0.3, 2: -0.9, 5: -0.9}.get(idx, -0.3)
                self.grip = 1.0
                self.phase = 'sweep'
                self.queue = self.plan_sweep(s)
            else:
                self.cand += 1
                self.grip = 1.0
                nq = self.plan_open(s, self.cand)
                if nq:
                    self.queue = nq
                else:
                    self.phase = 'sweep'
                    self.queue = self.plan_sweep(s)
        else:
            self.queue = self.plan_sweep(s)
