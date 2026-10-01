import numpy as np
from env_client import make_env

def feats(obs, name):
    o = obs.get_object_from_name(name)
    return {f: float(obs.get(o, f)) for f in ENV.observation_space.type_features[o.type]}

def rob(obs):
    o = obs.get_object_from_name('robot')
    return np.array([obs.get(o, k) for k in ('x', 'y', 'theta', 'arm_joint', 'vacuum')])

def blk(obs, name):
    o = obs.get_object_from_name(name)
    return np.array([obs.get(o, k) for k in ('x', 'y', 'theta')])

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def act(dx=0, dy=0, dth=0, darm=0, vac=0):
    return np.array([dx, dy, dth, darm, vac], dtype=np.float32)

def gripper_tip(r):
    """Tip of gripper: base + arm_joint along theta (approx; verify)."""
    x, y, th, arm = r[:4]
    return np.array([x + arm * np.cos(th), y + arm * np.sin(th)])

def block_corners(b, w=0.28, h=0.04):
    x, y, th = b
    c, s = np.cos(th), np.sin(th)
    loc = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    return np.array([[x + c * px - s * py, y + s * px + c * py] for px, py in loc])

def _servo(env, obs, tgt, idx, vac, max_steps, tol, log):
    lim = [.05, .05, .196, .1]
    n = 0; term = False
    for _ in range(max_steps):
        r = rob(obs)
        err = [tgt[0]-r[0], tgt[1]-r[1], wrap(tgt[2]-r[2]), tgt[3]-r[3]]
        err = [e if k in idx else 0.0 for k, e in enumerate(err)]
        if max(abs(e) for e in err) < tol:
            return obs, True, n, term
        a = act(*[float(np.clip(e, -l, l)) for e, l in zip(err, lim)], vac)
        obs, _, term, _, _ = env.step(a); n += 1
        if np.allclose(rob(obs)[:4], r[:4], atol=1e-8):
            if log: print('  stuck at', rob(obs), 'err', np.round(err, 4))
            return obs, False, n, term
        if term: return obs, True, n, term
    return obs, False, n, term

def drive_to(env, obs, tx, ty, tth, tarm=None, vac=0, max_steps=300, tol=1e-4, log=False):
    """Phased controller: retract arm (if needed), rotate, translate, extend arm."""
    r = rob(obs)
    tarm = r[3] if tarm is None else tarm
    tgt = [tx, ty, tth, tarm]; total = 0
    phases = ([[3]] if tarm < r[3] else []) + [[2], [0, 1], [3]]
    for idx in phases:
        obs, ok, n, term = _servo(env, obs, tgt, idx, vac, max_steps, tol, log)
        total += n
        if not ok or term: return obs, ok, total, term
    return obs, True, total, False

ENV = None
def new_env():
    global ENV
    ENV = make_env()
    return ENV

def creep(env, obs, dx=0., dy=0., dth=0., darm=0., vac=0, max_steps=400, min_frac=1/64):
    """Move in direction until contact; halves step on rejection. Returns obs, n_steps."""
    frac = 1.0; n = 0
    while n < max_steps and frac >= min_frac:
        r = rob(obs)
        o2, _, term, _, _ = env.step(act(dx*frac, dy*frac, dth*frac, darm*frac, vac)); n += 1
        if np.allclose(rob(o2)[:4], r[:4], atol=1e-8):
            frac /= 2
        obs = o2
        if term: break
    return obs, n

W, H = 0.28, 0.04
def block_center(b):
    x, y, th = b
    return np.array([x + np.cos(th)*W/2 - np.sin(th)*H/2, y + np.sin(th)*W/2 + np.cos(th)*H/2])

def side_grasp_pose(b, side=-1, arm=0.2, gap=0.01, along=0.0):
    """Robot pose to grasp block's long side. side=-1: face at corner side (-n), gripper points +n.
    along: offset along block long axis from center. Returns (x,y,theta)."""
    th = b[2]; u = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
    c = block_center(b) + along*u
    d = -side*n     # direction robot points
    rc = c - d*(H/2 + arm + 0.01 + gap)
    return rc[0], rc[1], float(np.arctan2(d[1], d[0]))

def approach_and_grasp(env, obs, name, side=-1, arm=0.2, along=0.0, standoff=0.12, log=False):
    """Drive to standoff pose, move straight along gripper direction, vacuum on."""
    x, y, t = side_grasp_pose(blk(obs, name), side, arm, 0.005, along)
    d = np.array([np.cos(t), np.sin(t)])
    obs, ok, n, _ = drive_to(env, obs, x - d[0]*standoff, y - d[1]*standoff, t, arm, log=log)
    if not ok: return obs, False
    obs, ok, n, _ = drive_to(env, obs, x, y, t, arm, log=log)
    obs, *_ = env.step(act(vac=1))
    return obs, ok
