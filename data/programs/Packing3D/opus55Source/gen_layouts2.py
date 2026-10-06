import json, math, sys
import numpy as np
from layout import *
from shapely.geometry import box
S=("cuboid",(0.05,0.05)); E=("tri",("equilateral",0.1,0.1)); R=("tri",("right",0.1,0.1))
D={"S":S,"E":E,"R":R}
PER={"S":math.pi/2,"E":2*math.pi/3,"R":2*math.pi}
YMAX=float(sys.argv[1]) if len(sys.argv)>1 else math.pi/2
keys=sys.argv[2].split(",") if len(sys.argv)>2 else ["R","RS","ERS","RRS","ERSS","RRSS"]
inner=box(*RACK_INNER)
def solve(key, rng):
    parts=[D[c] for c in key]; fps=[local_footprint(k,p) for k,p in parts]
    cents=[np.array(fp.centroid.coords[0]) for fp in fps]; n=len(parts)
    def to_origin(P):
        out=[]
        for k in range(n):
            cx,cy,yaw=P[k]; c,s=math.cos(yaw),math.sin(yaw); off=cents[k]
            out.append((cx-(c*off[0]-s*off[1]), cy-(s*off[0]+c*off[1]), yaw))
        return out
    def score(P):
        for k,ch in enumerate(key):
            if ch=="R" and abs(P[k][2])>YMAX: return -1e9
        return clearance([place_poly(fps[k],*o) for k,o in enumerate(to_origin(P))], inner)
    best=None
    for r in range(40):
        P=np.zeros((n,3)); P[:,0]=rng.uniform(0.23,0.37,n); P[:,1]=rng.uniform(-0.12,0.12,n)
        for k,ch in enumerate(key):
            P[k,2]=rng.uniform(-YMAX,YMAX) if ch=="R" else rng.uniform(0,PER[ch])
        s=score(P); step=np.array([0.02,0.02,0.3])
        for it in range(2500):
            Q=P.copy()
            if rng.random()<0.3: Q+=rng.normal(0,1,Q.shape)*step*0.5
            else: Q[rng.integers(n)]+=rng.normal(0,1,3)*step
            sq=score(Q)
            if sq>=s: P,s=Q,sq
            if it%500==499: step*=0.6
        if best is None or s>best[1]: best=(P.copy(),s)
    P,s=best
    step=np.array([0.003,0.003,0.03])
    for it in range(4000):
        Q=P.copy(); Q[rng.integers(n)]+=rng.normal(0,1,3)*step
        sq=score(Q)
        if sq>=s: P,s=Q,sq
    return to_origin(P), s
out={}
for key in keys:
    lay,c=solve(key, np.random.default_rng(0))
    print(key, round(c,4), [round(o[2],2) for o in lay], flush=True)
    out[key]=[list(map(float,o)) for o in lay]
json.dump(out, open(f"layouts_ymax_{YMAX:.2f}.json","w"), indent=1)
