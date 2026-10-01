from env_client import make_env
from approach import GeneratedApproach
env=make_env()
use=0; tot=0; costs=[]
for seed in range(200):
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    ap._read(obs)
    btns=list(ap.buttons)
    # table buttons
    tb=[b for b in btns if b[2]>1.39]
    if tb: use+=1
    tot+=1
print("episodes with button above y 1.39:",use,"/",tot)
