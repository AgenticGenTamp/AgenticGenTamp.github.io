import sys, time, numpy as np
sys.path.insert(0,'.')
from approach import GeneratedApproach
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[0,1], eval_counts=[0,1])
for seed in range(6):
    st, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(st, info)
    steps=0; pert = 2 + seed % 4
    while steps < 200:
        a = ap.get_action(st)
        if steps == pert: a = np.zeros(11, dtype=np.float32); a[0]=0.1
        st, r, term, trunc, _ = env.step(a); steps+=1
        if term: break
    print(f"seed {seed} pert@{pert} solved={term} steps={steps} replans={ap.replans}", flush=True)
