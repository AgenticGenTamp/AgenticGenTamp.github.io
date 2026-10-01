import sys, time, numpy as np
from env_client import make_env
import approach
from approach import *
seed = int(sys.argv[1])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for step in range(1000):
    a = ap.get_action(obs)
    prev_q = ap.q.copy()
    exp = ap.expected
    obs, r, term, trunc, info = env.step(a)
    ap._parse(obs)
    if exp is not None and np.abs(ap.q[:3]-exp[:3]).max() > 1e-3:
        print("step", step, "rejected action", a, "from", prev_q, "exp", exp)
        obst, names = ap.make_obstacles(exclude=(ap.held,) if ap.held else ())
        for m in [0.0, 0.002, 0.004]:
            mod = Model(obst, ap.held_local, margin=m)
            print(" margin", m, "model hits:", mod.hits(exp[None])[0], [names[i] for i in np.nonzero(mod.per_obj(exp[None])[0])[0]], "wall", mod.wall_bad(exp[None])[0])
        # nearby objects
        for n in names:
            c = ap.center(n)
            if np.linalg.norm(c-exp[:2]) < 0.45: print("  near", n, ap.rects[n])
        # probe: try same action with smaller rotation
        break
q0 = ap.q.copy()
print("q now", q0)
def rot_ok(dth):
    global obs
    o2, *_ = env.step(np.array([0,0,dth,0,0],dtype=np.float32))
    ap._parse(o2); ok = abs(wrap(ap.q[2]-q0[2]-dth))<1e-4
    if ok:
        env.step(np.array([0,0,-dth,0,0],dtype=np.float32))
    return ok
lo, hi = 0.0, -0.19
for _ in range(12):
    mid=(lo+hi)/2
    if rot_ok(mid): lo=mid
    else: hi=mid
print("max ok rotation", lo)
qt = q0.copy(); qt[2] = wrap(q0[2]+hi)
obst, names = ap.make_obstacles()
# distances: gripper poly vs objects, arm segment
mod = Model(obst, None, margin=0.0)
for m in [0.0, 0.005, 0.01, 0.02, 0.03]:
    print(m, [names[i] for i in np.nonzero(Model(obst,None,margin=m).per_obj(qt[None])[0])[0]])
# arm segment check: sample points from base center to gripper
c,s=np.cos(qt[2]),np.sin(qt[2])
for n in names:
    C = ap.corners(n)
    for t in np.linspace(0, qt[3], 21):
        p = qt[:2]+t*np.array([c,s])
        # point in polygon
        ob = Obstacles([C])
        if ob.circle_hit(p[None], 0.0, 0.0)[0]:
            print("arm point inside", n, "t=", t); break
