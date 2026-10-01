from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import time
import json
rng=np.random.default_rng(2026)
seeds=rng.integers(1,1000000,20).tolist()
for seed in seeds:
    env=make_env()
    state,info=env.reset(seed=seed)
    policy=GeneratedApproach(env.action_space,env.observation_space,{})
    policy.reset(state,info)
    started=time.monotonic()
    max_retries=0
    for step in range(env.max_steps):
        state,reward,terminated,truncated,info=env.step(policy.get_action(state))
        max_retries=max(max_retries,policy.retries)
        if terminated or truncated:
            break
    record=dict(seed=seed,terminated=bool(terminated),truncated=bool(truncated),steps=step+1,retries=max_retries,seconds=round(time.monotonic()-started,3),which=policy.which,phase=policy.phase)
    print(json.dumps(record),flush=True)
    if not terminated or truncated:
        with open('audit_failure_'+str(seed)+'.json','w') as out:
            json.dump(state.tolist(),out)
    env.close()
