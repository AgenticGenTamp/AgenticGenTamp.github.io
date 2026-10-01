import numpy as np, json
from expC_place2 import trial
out=[]
ts=[]
for d in [-0.30,-0.25,-0.20,-0.10,0.10,0.20,0.25,0.30]:
    ts.append((d,0,0.92,None,f"x{d}")); ts.append((0,d,0.92,None,f"y{d}"))
for t in ts:
    r=trial(*t)
    print(json.dumps({k:r[k] for k in ['tag','tool','blk_held','blk_final','term','ga_after','rej_carry','steps_grasp2rel'] if k in r}),flush=True)
    out.append(r)
json.dump(out,open("expC_r3.json","w"))
