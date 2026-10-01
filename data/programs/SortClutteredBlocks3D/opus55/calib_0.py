from env_client import make_env
import numpy as np
from kin import fk_arm
env=make_env(); obs,info=env.reset(seed=0)
for n in sorted(obs.get_object_names()):
    o=obs.get_object_from_name(n)
    print(n,o.type)
r=obs.get_object_from_name('robot')
F=['pos_base_x','pos_base_y','pos_base_rot']+[f'pos_arm_joint{i}' for i in range(1,8)]
v=[obs.get(r,f) for f in F]; print(np.round(v,4))
q=np.array(v[3:]); tip,R,ps,ax=fk_arm(q,0); print('fk bracelet',tip, R[:,2])
for n in sorted(obs.get_object_names()):
    o=obs.get_object_from_name(n)
    try: print(n,[round(obs.get(o,f),3) for f in ['x','y','z','qw','qz']])
    except Exception as e: pass
