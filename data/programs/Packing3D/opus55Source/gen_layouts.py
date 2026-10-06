import itertools, json, math
import numpy as np
from layout import *
S=("cuboid",(0.05,0.05)); E=("tri",("equilateral",0.1,0.1)); R=("tri",("right",0.1,0.1))
D={"S":S,"E":E,"R":R}
combos=["E","R","ES","RS","EES","ERS","RRS","EESS","ERSS","RRSS"]
out={}
for key in combos:
    parts=[D[c] for c in key]
    best=None
    for seed in range(4):
        # reorder so compute_layout's seeds apply; compute_layout handles any order
        res,c=compute_layout(parts, rng=np.random.default_rng(seed))
        fps=[local_footprint(k,p) for k,p in parts]
        # extra refinement from result: convert origin->centroid params
        cents=[]
        for (x,y,yaw),fp in zip(res,fps):
            cx,cy=fp.centroid.coords[0]; cs,sn=math.cos(yaw),math.sin(yaw)
            cents.append((x+cs*cx-sn*cy, y+sn*cx+cs*cy, yaw))
        res2,c2,_=refine_layout(fps,RACK_INNER,cents,iters=6000,rng=np.random.default_rng(100+seed),step0=(0.004,0.004,0.04))
        if c2>c: res,c=res2,c2
        if best is None or c>best[1]: best=(res,c)
    print(key, round(best[1],4), flush=True)
    out[key]=[list(map(float,r)) for r in best[0]]
with open("layouts_cache.py","w") as f:
    f.write("# Precomputed rack layouts (origin x, y, yaw) for standard part sets.\n")
    f.write("# Keys: sorted chars E=equilateral tri, R=right tri, S=0.1 square.\n")
    f.write("LAYOUTS = "+json.dumps(out, indent=1)+"\n")
