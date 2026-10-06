import sys, time, numpy as np
sys.path.insert(0,'.')
from approach import *
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[0], eval_counts=[0])
L=np.load('scratch/lib_all.npy')
good={}
for seed in range(0,40):
    st,_=env.reset(seed=seed)
    ap=GeneratedApproach(None,None,{}); sim=ap._get_sim(0); flat=sim.sync(st,set_walls=True)
    with sim.env.client():
        pl=Planner(sim,time.time()+60); q0=flat[:10].astype(float)
        c=pl.grasp_candidates(q0, sim.env.penned)
        for nn,q in c:
            if nn>5: break
            sim.env.set_state(flat)
            v=pl.valid_grasp_conf(q, sim.env.penned)
            i=np.argmin(np.abs(L[:,:7]-q[3:]).sum(1))
            good.setdefault(i,[0,0])[int(v)]+=1
            if v and seed<10: print(seed, np.round(q,2))
ids=[i for i,(b,o) in good.items() if o>0]
print("n5 entries", len(good), "valid", len(ids))
np.save('scratch/good5.npy', np.array(ids))
for i in ids[:20]:
    T=L[i,7:10]; print(i, good[i], "r=%.2f pa=%.2f z=%.3f phi=%.2f beta=%.2f"%(np.hypot(T[0],T[1]), np.arctan2(T[1],T[0]), T[2], L[i,10], L[i,11]))
