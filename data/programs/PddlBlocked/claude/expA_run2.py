import numpy as np, json
from expA_lib import *
env,obs,blk = new_rig()
out=[]
def run(name,cands):
    for (a,b,c) in cands:
        o,b0 = reset(env); o,r = trial(env,o,b0,a,b,c)
        r['sweep']=name; out.append(r); print(json.dumps(r), flush=True)
print("== A fine")
run("A",[(a,0.0,0.0) for a in [-0.01,0.01,0.045,0.05,0.055]])
print("== B fine")
run("B",[(0.03,b,0.0) for b in [-0.015,-0.01,-0.005,0.005,0.01,0.015]])
print("== C fine")
run("C",[(0.03,0.0,c) for c in [-0.05,-0.03,-0.01,0.01,0.03,0.05,0.07]])
print("== 2D")
run("2D",[(a,b,c) for a in [0.01,0.03] for b in [-0.005,0.0,0.005] for c in [0.0,0.03]])
json.dump(out, open("expA_r2.json","w")); env.close()
