import numpy as np, json, sys, time
from expA_lib import *
env,obs,blk = new_rig()
out=[]
def run(name, cands):
    global obs
    for (a,b,c) in cands:
        o,b0 = reset(env)
        o,r = trial(env,o,b0,a,b,c)
        r['sweep']=name; out.append(r); print(json.dumps(r), flush=True)
A=[round(x,3) for x in np.arange(-0.06,0.1001,0.02)]
B=[round(x,3) for x in np.arange(-0.06,0.0601,0.02)]
C=[round(x,3) for x in np.arange(-0.10,0.1001,0.02)]
run("A",[(a,0.0,0.0) for a in A])
run("B",[(0.03,b,0.0) for b in B])
run("C",[(0.03,0.0,c) for c in C])
json.dump(out, open("expA_r1.json","w")); env.close()
