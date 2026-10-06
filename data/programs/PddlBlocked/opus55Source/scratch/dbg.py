import sys, time, numpy as np
sys.path.insert(0,'.')
import approach
from approach import *
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[0,1,2,3], eval_counts=[0,1,2,3])
for seed in map(int, sys.argv[1:]):
    st, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.t0=time.time()
    n=ap._count(st); sim=ap._get_sim(n); flat=sim.sync(st, set_walls=True)
    with sim.env.client():
        pl = Planner(sim, time.time()+60)
        q0=flat[:10].astype(float)
        cands = pl.grasp_candidates(q0, sim.env.penned)
        print("seed", seed, "cand N hist", np.unique([c[0] for c in cands], return_counts=True))
        nvalid={}
        for nn,q in cands:
            if nn>8: break
            sim.env.set_state(flat)
            if pl.valid_grasp_conf(q, sim.env.penned): nvalid[nn]=nvalid.get(nn,0)+1
        print(" valid", nvalid)
        gr = pl.plan_grasp(flat, sim.env.penned, max_keep=10)
        print(" grasps N", [g[0] for g in gr])
        rng=np.random.default_rng(0)
        for g in gr[:5]:
            r = pl.plan_place_on_plate(g[3], rng, max_n=3)
            print("  carry", None if r is None else r[0])
