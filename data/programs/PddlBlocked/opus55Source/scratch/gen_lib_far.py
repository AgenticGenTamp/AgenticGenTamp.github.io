import sys, math, time, numpy as np
sys.path.insert(0,'.')
from approach import *
from approach import _Rz, _Ry, _wrap, _rot_err
sim = Sim(0); e = sim.env
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
N = int(sys.argv[2]) if len(sys.argv)>2 else 500
out=[]
t=time.time()
with e.client():
    q0 = sim.get_conf()
    base = np.zeros(3)
    tries=0
    while len(out)<N:
        tries+=1
        r = rng.uniform(0.75, 1.0); pa = rng.uniform(-0.7, 1.4)
        phi = rng.uniform(-1.1, 1.6); beta = math.radians(rng.uniform(14, 24.5))
        z = rng.uniform(0.835, 0.877)
        T = np.array([r*math.cos(pa), r*math.sin(pa), z])
        Rt = _Rz(phi) @ _Ry(beta)
        qi = q0.copy(); qi[:3]=0
        if rng.random()<0.8:
            qi[3:] = q0[3:] + rng.normal(0, 0.6, 7)
        else:
            qi[3:] = rng.uniform(sim.lo[3:], sim.hi[3:])
        qi = np.clip(qi, sim.lo, sim.hi)
        q, ok = sim.ik(T, Rt, qi, q_ref=qi, wbase=0.0, iters=80)
        if not ok: continue
        out.append(np.concatenate([q[3:], T, [phi, beta]]))
print(len(out), tries, time.time()-t, file=sys.stderr)
np.save(f'scratch/libfar_{sys.argv[1]}.npy', np.array(out))
