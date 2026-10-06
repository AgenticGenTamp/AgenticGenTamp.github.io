import sys, time, numpy as np
sys.path.insert(0,'.')
from approach import GeneratedApproach
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[1,2], eval_counts=[1,2])
mode=sys.argv[1]
for seed in range(int(sys.argv[2]), int(sys.argv[3])):
    st, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap._force=mode
    t=time.time(); ap.reset(st, info); tp=time.time()-t
    steps=0
    while steps < 400:
        st, r, term, trunc, _ = env.step(ap.get_action(st)); steps+=1
        if term or trunc: break
    print(f"seed {seed} solved={term} steps={steps} plan_t={tp:.1f} planlen={len(ap.plan)}", flush=True)
