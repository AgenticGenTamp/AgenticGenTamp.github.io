import time
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for seed in [0,2,4]:
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    t=time.time(); ap.get_action(obs); t1=time.time()-t
    ap._read(obs); t=time.time(); ap._make_plan(); print(seed, len(ap.buttons), round(t1,3), round(time.time()-t,3))
