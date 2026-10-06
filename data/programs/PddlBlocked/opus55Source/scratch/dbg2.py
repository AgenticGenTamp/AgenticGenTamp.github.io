import sys, time, numpy as np
sys.path.insert(0,'.')
from approach import *
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[0], eval_counts=[0])
for seed in map(int, sys.argv[1:]):
    st,_=env.reset(seed=seed)
    ap=GeneratedApproach(None,None,{}); sim=ap._get_sim(0); flat=sim.sync(st,set_walls=True)
    e=sim.env
    with e.client():
        pl=Planner(sim,time.time()+60); q0=flat[:10].astype(float)
        c=pl.grasp_candidates(q0, e.penned, ox_list=(0.0,0.01,0.02,0.03,0.04))
        n5=[q for nn,q in c if nn<=5]
        val=[]
        for q in n5:
            e.set_state(flat)
            if pl.valid_grasp_conf(q, e.penned): val.append(q)
        print("seed", seed, "pen", np.round(e.sim_pose if False else flat[19+30:19+30+3],3), "n5", len(n5), "valid", len(val))
        gr = pl.plan_grasp(flat, e.penned, max_keep=20, extra=(0,))
        gr=[g for g in gr if g[0]<=5]
        print(" grasp plans 5:", len(gr))
        rng=np.random.default_rng(0)
        for g in gr[:6]:
            r = pl.plan_place_on_plate(g[3], rng, max_n=1, samples=5000)
            print("  carry1", r is not None)
        for q in val:
            confs=pl.interp(q0,q,5)
            for k,c in enumerate(confs):
                e.set_state(flat); sim.set_conf(c)
                if e._in_collision(): print("  coll at step", k+1); break
            else: print("  path free")
        for q in val:
            confs=pl.interp(q0,q,5)
            ok, fl, _ = pl.simulate(flat, confs, [0]*4+[-1])
            print("  sim", ok, None if fl is None else fl[11], np.max(np.abs(conf_delta(q0,q,sim.circ))))
