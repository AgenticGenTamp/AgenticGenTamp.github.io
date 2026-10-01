import sys, numpy as np
from env_client import make_env
from approach import *
seed = int(sys.argv[1])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for step in range(1000):
    a = ap.get_action(obs)
    qb = ap.q.copy(); held = ap.held
    obs, r, term, trunc, info = env.step(a)
    if term: print("solved"); sys.exit()
    ap._parse(obs)
    if np.abs(ap.q - qb).max() < 1e-9 and np.abs(a[:4]).max() > 1e-6: break
print("rejected at", step, "a", a, "q", qb, "held", held)
q0 = qb.copy()
def try_act(act):
    o2, *_ = env.step(np.array(list(act)+[0], dtype=np.float32))
    ap._parse(o2); moved = np.abs(ap.q-q0).max() > 1e-9
    if moved:
        env.step(np.array([-act[0],-act[1],-act[2],-act[3],0], dtype=np.float32)); ap._parse(env.step(np.zeros(5,dtype=np.float32))[0])
        assert np.abs(ap.q-q0).max()<1e-6, ap.q
    return moved
print("rot only ok:", try_act([0,0,a[2],0]), "xy only ok:", try_act([a[0],a[1],0,0]), "arm only:", try_act([0,0,0,a[3]]))
lo, hi = 0.0, float(a[2])
for _ in range(14):
    mid=(lo+hi)/2
    if try_act([0,0,mid,0]): lo=mid
    else: hi=mid
print("rotation threshold", lo)
qt = q0.copy(); qt[2] = wrap(q0[2]+hi)
obst, names = ap.make_obstacles()
for m in [0.0, 0.005, 0.01, 0.02]:
    print(m, [names[i] for i in np.nonzero(Model(obst,None,margin=m).per_obj(qt[None])[0])[0]])
gp = Model(obst,None).body_polys(qt[None])[0][0]
print("gripper corners", gp.round(4))
for n in names:
    C = ap.corners(n)
    if np.linalg.norm(C.mean(0)-qt[:2])<0.4: print(n, C.round(4).tolist())
