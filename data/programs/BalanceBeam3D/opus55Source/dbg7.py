import numpy as np, math
from approach import Kin, HOME, wrap
k=Kin()
def rot_tilt(alpha):
    # fingers along base y; approach axis z tilted toward +x by alpha (pointing down and forward)
    x=np.array([0,1.0,0])
    z=np.array([math.sin(alpha),0,-math.cos(alpha)])
    y=np.cross(z,x)
    return np.stack([x,y,z],1)
rng=np.random.default_rng(0)
for alpha in [0, 0.3, 0.6, 0.9, 1.2]:
    best=None
    for r in [0.4,0.5,0.6,0.7]:
        for t in range(6):
            sd = HOME if t==0 else HOME+rng.normal(0,0.5,7)
            q,pe,re_=k.ik(np.zeros(3), np.array([r,0,0.033]), rot_tilt(alpha), sd, iters=300, bias=HOME)
            if pe<2e-3 and re_<5e-3:
                c=np.max(np.abs([wrap(x) for x in q-HOME]))
                if best is None or c<best[0]: best=(c,r,np.round(q,2))
    print(alpha, best)
