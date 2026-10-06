import sys, time, numpy as np
sys.path.insert(0,'.')
from approach import *
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[0], eval_counts=[0])
L=np.load('scratch/lib_all.npy')
good={}
for seed in range(0,30):
    st,_=env.reset(seed=seed)
    ap=GeneratedApproach(None,None,{}); sim=ap._get_sim(0); flat=sim.sync(st,set_walls=True)
    with sim.env.client():
        pl=Planner(sim,time.time()+60); q0=flat[:10].astype(float)
        c=pl.grasp_candidates(q0, sim.env.penned)
        for nn,q in c:
            if nn>6: break
            sim.env.set_state(flat)
            v=pl.valid_grasp_conf(q, sim.env.penned)
            # find lib idx
            i=np.argmin(np.abs(L[:,:7]-q[3:]).sum(1))
            good.setdefault(i,[0,0])[int(v)]+=1
for i,(bad,ok) in sorted(good.items(), key=lambda x:-x[1][1]):
    T=L[i,7:10]; print(i, bad, ok, "r=%.2f pa=%.2f z=%.3f phi=%.2f beta=%.2f"%(np.hypot(T[0],T[1]), np.arctan2(T[1],T[0]), T[2], L[i,10], L[i,11]), np.round(L[i,:7],2))
