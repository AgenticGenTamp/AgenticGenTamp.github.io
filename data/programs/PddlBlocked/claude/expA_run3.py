import numpy as np, json
from expA_lib import *
env,obs,blk = new_rig()
def run(cands):
    for (a,b,c) in cands:
        o,b0 = reset(env); o,r = trial(env,o,b0,a,b,c)
        print(r['a'],r['b'],r['c'],'ga=',r.get('ga'),r['status'],r.get('rej_sub'),r.get('reach_err'), flush=True)
run([(0.03,0.0,0.01)]*3)
run([(a,0.0,0.0) for a in [0.035,0.04,0.04,0.042,0.045]])
run([(0.03,0.0,c) for c in [0.07,0.075,0.08,0.08]])
run([(0.03,0.0,c) for c in [-0.02,-0.025,-0.03]])
run([(0.03,b,0.0) for b in [-0.012,0.007,0.008,0.01]])
env.close()
