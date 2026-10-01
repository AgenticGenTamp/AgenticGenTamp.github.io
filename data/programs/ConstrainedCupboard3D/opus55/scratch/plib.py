import sys, time; sys.path.insert(0,'/sandbox'); sys.path.insert(0,'/sandbox/scratch')
from env_client import make_env
from ctrl import *
from kin import ik, fk

LOG = []  # non -1 rewards
def watch(S, tag=''):
    pass

class P:
    def __init__(self, seed, k=None, rod=None):
        self.env = make_env()
        obs, info = self.env.reset(seed=seed, options={} if k is None else {'object_count': k})
        self.S = Sim(self.env, obs); self.info = info
        F = self.env.observation_space.get_type('mujoco_fixture')
        self.cols = {o.name: obs.get(o, 'y') for o in obs.get_objects(F)}
        self.nr = 0
    def chk(self):
        S = self.S
        for i in range(self.nr, len(S.rew)):
            if S.rew[i] != -1.0:
                LOG.append((i, S.rew[i], {k: np.round(v, 3).tolist() for k, v in S.rods().items()}))
                print('REWARD', i, S.rew[i], LOG[-1][2], flush=True)
        self.nr = len(S.rew)
    def pick(self, name):
        ok = pick_rod(self.S, name); self.name = name; self.chk(); return ok
    def rel(self):
        """rod center minus grasp point, and rod axis x-component"""
        S = self.S; r = S.rods()[self.name]; gp = grasp_point_world(S)
        w, x, y, z = r[3:7]
        ax = np.array([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)])  # local y axis
        return r[:3] - gp, ax
    def moveto(self, p, R, bt=None, steps=100, tol=0.004):
        S = self.S
        b = S.base() if bt is None else np.asarray(bt, float)
        qt, e1, e2 = ik(S.q(), world_to_arm(b, np.asarray(p, float)), R_world_to_arm(b, R), tool=TOOL)
        n = S.goto(qt=qt, bt=bt, steps=steps, tol=tol); self.chk()
        return e1, n, np.abs(qt - S.q()).max()
    def push(self, y, z, x0, x1, R, dx=0.01, sub=8, thr=0.012):
        """move grasp point along +x at (y,z) from x0 to x1; return first x where blocked (gp x)."""
        S = self.S
        moveto_slow(self, [x0, y, z], R, steps=150, vmax=0.03)
        d0, a0 = self.rel()
        x = x0
        while x < x1:
            x += dx
            b = S.base()
            qt, e1, _ = ik(S.q(), world_to_arm(b, np.array([x, y, z])), R_world_to_arm(b, R), tool=TOOL)
            goto_slow(S, qt=qt, steps=sub, tol=0.002, vmax=0.03)
            gp = grasp_point_world(S); d, a = self.rel()
            lag = x - gp[0]; slip = max(np.linalg.norm(d - d0), 0.15*np.linalg.norm(a - a0))
            if lag > thr or slip > thr or abs(gp[1]-y) > thr or abs(gp[2]-z) > thr:
                self.chk()
                res = (round(x, 3), np.round(gp, 3).tolist(), round(slip, 3))
                # back off
                moveto_slow(self, [x0, y, z], R, steps=150, vmax=0.03)
                return res
        self.chk()
        self.moveto([x0, y, z], R, steps=60)
        return None

def goto_slow(S, qt=None, bt=None, steps=200, tol=0.004, vmax=0.04):
    for t in range(steps):
        a = np.zeros(11); a[10] = S.grip; done = True
        if qt is not None:
            e = qt - S.q(); e[[0,2,4,6]] = (e[[0,2,4,6]] + np.pi) % (2*np.pi) - np.pi
            a[3:10] = np.clip(e, -vmax, vmax); done &= np.abs(e).max() < tol
        if bt is not None:
            b = S.base(); e = bt - b; e[2] = (e[2] + np.pi) % (2*np.pi) - np.pi
            a[0:3] = np.clip(e, -vmax, vmax); done &= np.abs(e).max() < tol
        if done and t > 0: return t
        S.step(a)
    return steps

def moveto_slow(p, pw, R, bt=None, steps=250, vmax=0.04):
    S = p.S; b = S.base() if bt is None else np.asarray(bt, float)
    qt, e1, e2 = ik(S.q(), world_to_arm(b, np.asarray(pw, float)), R_world_to_arm(b, R), tool=TOOL)
    n = goto_slow(S, qt=qt, bt=bt, steps=steps, vmax=vmax); p.chk(); return e1, n

RH = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.]])  # ee z=+x world, ee x=+y world, fingers close along world z

def ik_multi(S, pw, R, b=None, tries=12, rng=np.random.default_rng(0)):
    b = S.base() if b is None else np.asarray(b, float)
    pa = world_to_arm(b, np.asarray(pw, float)); Ra = R_world_to_arm(b, R)
    best = None
    seeds = [S.q(), np.array([0, 0.3, np.pi, -1.8, 0, -0.9, np.pi/2]), np.array([0, 0.6, 0, -1.8, 0, -0.9, 0])]
    seeds += [S.q() + rng.normal(0, 0.8, 7) for _ in range(tries)]
    for s0 in seeds:
        q, e1, e2 = ik(s0, pa, Ra, tool=TOOL, iters=300)
        if e1 < 1e-3 and e2 < 1e-2:
            d = (q - S.q()); d[[0,2,4,6]] = (d[[0,2,4,6]] + np.pi) % (2*np.pi) - np.pi
            c = np.abs(d).sum()
            if best is None or c < best[0]: best = (c, q)
    return None if best is None else best[1]

def probe_dir(p, start, direction, dist, R, dx=0.004, thr=0.006, vmax=0.03):
    """move grasp point from start along unit direction up to dist; return grasp-point at contact (or None)."""
    S = p.S; start = np.asarray(start, float); direction = np.asarray(direction, float)
    qt = ik_multi(S, start, R)
    if qt is None: return 'noik'
    goto_slow(S, qt=qt, steps=300, vmax=0.05)
    g0 = grasp_point_world(S)
    if np.linalg.norm(g0 - start) > 0.01: return ('start_fail', np.round(g0, 3).tolist())
    s = 0.0; b = S.base()
    while s < dist:
        s += dx; tgt = start + s * direction
        qt, e1, _ = ik(S.q(), world_to_arm(b, tgt), R_world_to_arm(b, R), tool=TOOL)
        goto_slow(S, qt=qt, steps=8, tol=0.001, vmax=vmax)
        g = grasp_point_world(S)
        if np.linalg.norm(g - tgt) > thr:
            p.chk()
            res = g
            qt = ik_multi(S, start, R); goto_slow(S, qt=qt, steps=200, vmax=0.05)
            return np.round(res, 4)
    p.chk(); qt = ik_multi(S, start, R); goto_slow(S, qt=qt, steps=200, vmax=0.05)
    return None
