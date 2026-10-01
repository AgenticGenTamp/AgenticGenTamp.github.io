import numpy as np, json
from expC_place2 import trial
ts=[(0,0.32,0.92,None,"y0.32"),(0,0.35,0.92,None,"y0.35"),(0,0.40,0.92,None,"y0.40"),(0,0.50,0.92,None,"y0.50"),
    (-0.32,0,0.92,None,"x-0.32"),(-0.35,0,0.92,None,"x-0.35"),(-0.40,0,0.92,None,"x-0.40"),
    (0,-0.32,0.92,None,"y-0.32"),(0,-0.35,0.92,None,"y-0.35")]
out=[]
for t in ts:
    r=trial(*t)
    print(json.dumps({k:r[k] for k in ['tag','blk_held','blk_final','ga_after','rej_carry'] if k in r}),flush=True)
    out.append(r)
json.dump(out,open("expC_r4.json","w"))
