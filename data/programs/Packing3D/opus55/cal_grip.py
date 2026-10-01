from cal_util import *
d=Driver(0)
for g in [1.0,1.0,-1.0,-1.0,0,1.0]:
    d.step(act(grip=g)); r=d.r; print(g, r['finger'], r['ga'], r['gtf'])
