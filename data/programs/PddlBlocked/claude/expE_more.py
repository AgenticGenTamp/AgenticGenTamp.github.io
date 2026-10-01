import numpy as np
from env_client import make_env
from farroute import FarRoute, plan_far_route, state_blocks, block_yaws, robot_state
env=make_env(); res={}
for seed in [4,5,6,7,8]:
    obs,info=env.reset(seed=seed)
    if info["object_count"]==0: res[seed]="no spares"; continue
    pol=FarRoute()
    try: pol.reset(obs,info)
    except Exception as e: res[seed]="planfail %s"%e; continue
    n=0; term=False
    for t in range(1000):
        obs,rew,term,trunc,info=env.step(pol.get_action(obs)); n+=1
        if term or trunc: break
    res[seed]=(n,bool(term),pol.phase)
print("EXTRA",res)
obs,info=env.reset(seed=0)
wps=plan_far_route(state_blocks(obs), robot_state(obs), yaws=block_yaws(obs))
print("plan_far_route ->",len(wps),"waypoints:",[w["note"] for w in wps])
env.close()
