import numpy as np, sys
from env_client import make_env
import approach as A

def trial(zg, seed=0, steps=200):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
    ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.z_grasp=zg
    ap.reset(obs,info)
    c0=A.obj_pos(obs,'cube1').copy()
    best=0; gwc=None
    for t in range(steps):
        a=ap.get_action(obs)
        if ap.phase=='close' and gwc is None:
            b,q,g=A.robot_state(obs); gwc=ap.gripper_world(b,q)[0]
        obs,r,te,tr,i=env.step(a)
        c=A.obj_pos(obs,'cube1'); best=max(best, float(c[2]))
        if ap.idx>0: break
    env.close()
    return round(best,4), np.round(gwc,4) if gwc is not None else None
for zg in [float(x) for x in sys.argv[1:]]:
    print(zg, trial(zg))
