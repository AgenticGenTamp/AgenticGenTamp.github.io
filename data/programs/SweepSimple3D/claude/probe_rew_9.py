import sys, numpy as np
from env_client import make_env
seeds=[int(s) for s in sys.argv[1].split(',')]
env=make_env(); os_=env.observation_space; T=os_.get_type
try:
    obs,info=env.reset(seed=seeds[0])
    a=np.zeros(11,dtype=np.float32)
    print("collision_helper:", env.check_action_collision(a))
except Exception as e:
    print("collision_helper FAIL", repr(e)[:120])
for sd in seeds:
    obs,info=env.reset(seed=sd)
    cs=[]; wp=None
    for o in obs.get_objects(T("mujoco_movable_object")):
        p=(round(float(obs.get(o,'x')),3),round(float(obs.get(o,'y')),3))
        if o.name.startswith('cube'): cs.append(p)
        else:
            q=[round(float(obs.get(o,f)),3) for f in ('qw','qx','qy','qz')]
            wp=(p,q)
    r=obs.get_objects(T("mujoco_tidybot_robot"))[0]
    rb=(round(float(obs.get(r,'pos_base_x')),3),round(float(obs.get(r,'pos_base_y')),3),round(float(obs.get(r,'pos_base_rot')),3))
    print("seed",sd,"n",info.get('object_count'),"cubes",cs,"wiper",wp,"base",rb,flush=True)
env.close()
