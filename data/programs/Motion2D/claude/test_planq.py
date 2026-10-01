import sys, math, numpy as np
from env_client import make_env
import approach
from approach import GeneratedApproach, _cheb
def plen(p): return sum(_cheb(*p[i],*p[i+1]) for i in range(len(p)-1))
env=make_env()
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap2=GeneratedApproach(env.action_space, env.observation_space, {})
# monkeypatch lattice spacing for ap2 via subclass
import types
orig_lat = GeneratedApproach._lattice
def fine_lat(self, start, r):
    xs, ys = orig_lat(self, start, r)
    wx0,wy0,wx1,wy1=self._bounds
    xs=set(xs); ys=set(ys)
    v=wx0+r
    while v<=wx1-r: xs.add(round(v,6)); v+=0.02
    v=wy0+r
    while v<=wy1-r: ys.add(round(v,6)); v+=0.02
    return sorted(xs), sorted(ys)
ap2._lattice = types.MethodType(fine_lat, ap2)
def make_plan_fine(self, start, th0):
    prim = 1.0 if math.cos(th0) >= -1e-9 else -1.0
    for gd in (prim,-prim):
        th = 0.0 if gd>0 else math.pi
        for m in (0.0005,0.0):
            p=self._plan(start, self.radius+m, th)
            if p: self.gdir=gd; self.margin=m; return p
    return None
ap2._make_plan = types.MethodType(make_plan_fine, ap2)
d=[]
for seed in range(int(sys.argv[2]), int(sys.argv[2])+int(sys.argv[1])):
    obs,info=env.reset(seed=seed)
    ap.reset(obs,info); ap2.reset(obs,info)
    a=plen(ap.path)/0.05; b=plen(ap2.path)/0.05
    d.append(a-b)
    if a-b>0.5: print("seed",seed,round(a,2),round(b,2),flush=True)
print("mean excess steps vs fine plan", round(float(np.mean(d)),3), "max", round(max(d),2))
env.close()
