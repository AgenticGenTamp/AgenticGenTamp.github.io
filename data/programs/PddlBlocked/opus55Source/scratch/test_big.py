import sys, time, numpy as np
from approach import GeneratedApproach
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[5,8], eval_counts=[5,8])
seeds = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv)>2 else range(5)
res=[]
for seed in seeds:
    st, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    t=time.time(); ap.reset(st, info); tp=time.time()-t
    steps=0; term=False
    while steps < 400:
        st, r, term, trunc, _ = env.step(ap.get_action(st)); steps+=1
        if term or trunc: break
    res.append(steps if term else None)
    print(f"seed {seed} solved={term} steps={steps} plan_t={tp:.1f} dbg={getattr(ap,'debug',None)}", flush=True)
ok=[r for r in res if r is not None]
print("solved", len(ok), "/", len(res), "mean steps", np.mean(ok) if ok else None)
