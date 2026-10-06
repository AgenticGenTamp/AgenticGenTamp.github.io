import sys, time, numpy as np
from approach import GeneratedApproach, obs_to_flat, Planner
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
k=int(sys.argv[1]); flat=np.load(sys.argv[2])[int(sys.argv[3])]
env=E([k],[k]); o,info=env.reset(seed=0, options={'object_count':k})
ap=GeneratedApproach(env.action_space, env.observation_space, None); ap.reset(o,info)
P=ap.planner
orig=P.simulate
def sim(flat_, acts):
    P.last_fail=None
    r=orig(flat_, acts)
    print("   sim", len(acts), r[0], P.last_fail, None if r[1] is None else [int(r[1][39+11*i+7]) for i in range(k)])
    return r
P.simulate=sim
ob=P.best_path
def bp(flat_, qc, qt, gl, check, pre=None, best_cost=1e9, max_tries=40):
    print('cand base', np.round(qt[:3],2))
    for c,_,seq in P.path_seqs(qc, qt, pre)[:12]:
        print('  seq', c, [np.round(w[:3],2).tolist() for w in seq[:-1]])
    return ob(flat_, qc, qt, gl, check, pre, best_cost, 12)
P.best_path=bp
print('base', np.round(flat[:3],2), 'held', flat[11])
r=ap.replan(flat)
print('----')
qc=flat[:10].copy()
P.simulate=orig
def bp2(flat_, qc, qt, gl, check, pre=None, best_cost=1e9, max_tries=40):
    for c,_,seq in P.path_seqs(qc, qt, pre)[4:8]:
        print('  seq', c, [(np.round(w[:3],2).tolist(), P.wp_ok(w), P.last_fail if False else None) for w in seq])
    raise SystemExit
P.best_path=bp2
ap.replan(flat)
