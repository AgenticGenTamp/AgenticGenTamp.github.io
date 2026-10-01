import numpy as np, time, sys
sys.path.insert(0,'.')
import approach as A
q0=np.array([0,-0.349,3.1416,-2.548,0,-0.873,1.571])
t0=time.time()
for i in range(50):
    q,err=A.ik(np.array([0.66,0.0,0.34]), A.rot_down(1.5708), q0, 0.12, iters=40)
print('ik40 per call ms', (time.time()-t0)/50*1000, 'err',err)
t0=time.time()
for i in range(200):
    q2,err=A.ik(np.array([0.66,0.0,0.34]), A.rot_down(1.5708), q, 0.12, iters=12)
print('ik12 warm per call ms', (time.time()-t0)/200*1000,'err',err)
print('home->target max joint travel', np.abs(q-q0).max(), np.round(q,3))
# try alternate seeds to reduce travel
best=None
rng=np.random.default_rng(0)
for k in range(60):
    seed = q0 + rng.normal(0,1.2,7)
    qq,e = A.ik(np.array([0.66,0.0,0.34]), A.rot_down(1.5708), seed, 0.12, iters=200)
    if e<0.002:
        d=np.abs(qq-q0).max()
        if best is None or d<best[0]: best=(d,qq)
print('best alt travel', best[0], np.round(best[1],3))
