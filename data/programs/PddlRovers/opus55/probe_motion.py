"""Helpers for empirically probing rover motion / collision geometry."""
import numpy as np
from env_client import make_env

env = make_env()
SP = env.observation_space
RT = SP.get_type('rover')
OT = SP.get_type('obstacle')


def reset(seed=0):
    global obs
    obs, _ = env.reset(seed=seed)
    return obs


def rov(i, o=None):
    o = o if o is not None else obs
    r = [x for x in o.get_objects(RT) if x.name == f'rover{i}'][0]
    return np.array([o.get(r, 'x'), o.get(r, 'y'), o.get(r, 'theta')])


def obstacles(o=None):
    o = o if o is not None else obs
    return {b.name: [round(float(o.get(b, f)), 3) for f in ('x', 'y', 'half_x', 'half_y')]
            for b in o.get_objects(OT)}


def step(i, dx=0., dy=0., dth=0.):
    """Move rover i by (dx,dy,dth); returns (moved?, new pose)."""
    global obs
    a = np.zeros(8, dtype=np.float32)
    a[4 * i:4 * i + 4] = [dx, dy, dth, 1.0]
    before = rov(i)
    obs, r, te, tr, info = env.step(a)
    after = rov(i)
    return not np.allclose(before[:2], after[:2], atol=1e-9) or abs(after[2]-before[2]) > 1e-9, after


def goto(i, x, y, maxstep=0.2, tries=400):
    """Greedy straight-line drive (axis-separated fallback). Returns final pose."""
    for _ in range(tries):
        p = rov(i)
        d = np.array([x, y]) - p[:2]
        if np.linalg.norm(d) < 1e-6:
            break
        s = np.clip(d, -maxstep, maxstep)
        ok, _ = step(i, s[0], s[1])
        if not ok:
            ok, _ = step(i, s[0], 0)
            if not ok:
                ok, _ = step(i, 0, s[1])
                if not ok:
                    break
    return rov(i)


def push(i, dx, dy, min_step=1e-4):
    """Move in direction (dx,dy) as far as possible, halving step on collision."""
    v = np.array([dx, dy], float)
    v = v / np.linalg.norm(v) * 0.2
    while np.linalg.norm(v) >= min_step:
        ok, _ = step(i, v[0], v[1])
        if not ok:
            v = v / 2
    return rov(i)


def wall_scan(seed=0, i=0, ys=None, xstart=0.45):
    """For each y, drive rover i to (xstart,y) then push toward -x; record min x."""
    reset(seed)
    sgn = 1 if i == 0 else -1
    ys = ys if ys is not None else np.round(np.arange(-2.25, 2.26, 0.1), 3)
    out = []
    for y in ys:
        p = goto(i, sgn * xstart, y)
        if abs(p[1] - y) > 1e-3 or abs(p[0] - sgn * xstart) > 1e-3:
            out.append((y, None, tuple(np.round(p[:2], 3))))
            continue
        q = push(i, -sgn, 0, 1e-5)
        out.append((y, round(float(q[0]), 4), None))
    return out


def goto_robust(i, x, y, lanes=(1.0, 1.6, 2.2, 0.6)):
    """Try direct goto; on failure route via vertical 'lanes' at x=+-lane."""
    sgn = 1 if i == 0 else -1
    p = goto(i, x, y)
    if np.allclose(p[:2], [x, y], atol=1e-3):
        return p
    for lx in lanes:
        lx *= sgn
        goto(i, lx, rov(i)[1])
        goto(i, lx, y)
        p = goto(i, x, y)
        if np.allclose(p[:2], [x, y], atol=1e-3):
            return p
    return p


def wall_scan2(seed=0, i=0, ys=None, xstart=0.3):
    reset(seed)
    sgn = 1 if i == 0 else -1
    ys = ys if ys is not None else np.round(np.arange(-2.25, 2.26, 0.1), 3)
    out = []
    for y in ys:
        p = goto_robust(i, sgn * xstart, y)
        if not np.allclose(p[:2], [sgn * xstart, y], atol=1e-3):
            out.append((y, None, tuple(np.round(p[:2], 3))))
            continue
        q = push(i, -sgn, 0, 1e-5)
        out.append((y, round(float(q[0]), 4), None))
        if sgn * q[0] < 0.1:  # crossed! come back
            goto(i, sgn * xstart, y)
    return out


def set_theta(i, th):
    import math
    for _ in range(50):
        d = (th - rov(i)[2] + math.pi) % (2 * math.pi) - math.pi
        if abs(d) < 1e-6:
            break
        step(i, 0, 0, max(-0.4, min(0.4, d)))
    return rov(i)


def obstacle_exact(name, o=None):
    o = o if o is not None else obs
    b = [x for x in o.get_objects(OT) if x.name == name][0]
    return np.array([o.get(b, f) for f in ('x', 'y', 'half_x', 'half_y')], float)


def contact_profile(seed, i, name, angles, theta=None, R0=0.6):
    """Approach obstacle center from each angle (direction from center to start); return contact center distances."""
    reset(seed)
    c = obstacle_exact(name)
    res = []
    for a in angles:
        s = c[:2] + R0 * np.array([np.cos(a), np.sin(a)])
        p = goto_robust(i, *s)
        if not np.allclose(p[:2], s, atol=1e-3):
            res.append((a, None)); continue
        if theta is not None:
            set_theta(i, theta)
        q = push(i, *(c[:2] - s), 1e-6)
        res.append((a, q[:2] - c[:2]))
    return res


def slide_scan(seed, i=0, x0=0.2275, dy=0.02, probe=0.01):
    """Slide rover i along the wall at |x|=x0 from bottom to top; at each y try stepping toward wall.
    Returns list of (y, event) where event in {'gap', 'blocked_up'}; plus final y."""
    reset(seed)
    sgn = 1 if i == 0 else -1
    p = goto_robust(i, sgn * 0.4, -2.2)
    p = goto(i, sgn * x0, -2.2)
    p = push(i, 0, -1, 1e-5)
    events = []
    while True:
        y = rov(i)[1]
        ok, _ = step(i, -sgn * probe, 0)
        if ok:
            q = push(i, -sgn, 0, 1e-5)
            events.append((round(float(y), 3), 'gap', round(float(q[0]), 3)))
            goto(i, sgn * x0, y)
        ok, p = step(i, 0, dy)
        if not ok:
            events.append((round(float(y), 3), 'blocked_up'))
            break
    return events


def slide_scan_full(seed, i=0, x0=0.2275, dy=0.02, probe=0.01, ytop=2.27):
    reset(seed)
    sgn = 1 if i == 0 else -1
    goto_robust(i, sgn * 0.4, -2.2)
    goto(i, sgn * x0, -2.2)
    push(i, 0, -1, 1e-5)
    events = []
    y_lo = rov(i)[1]
    while True:
        y = rov(i)[1]
        ok, _ = step(i, -sgn * probe, 0)
        if ok:
            q = push(i, -sgn, 0, 1e-5)
            events.append(('gap', round(float(y), 3), round(float(q[0]), 3)))
            goto(i, sgn * x0, y)
        ok, p = step(i, 0, dy)
        if not ok:
            # detour: find next reachable y along the wall
            yb = y
            yn = y + 0.05
            while yn < ytop:
                p = goto_robust(i, sgn * x0, yn)
                if np.allclose(p[:2], [sgn * x0, yn], atol=1e-3):
                    break
                yn += 0.05
            else:
                events.append(('scanned', round(float(y_lo), 3), round(float(yb), 3)))
                break
            # refine lower end of the new segment
            push(i, 0, -1, 1e-5)
            events.append(('scanned', round(float(y_lo), 3), round(float(yb), 3)))
            y_lo = rov(i)[1]
    return events
