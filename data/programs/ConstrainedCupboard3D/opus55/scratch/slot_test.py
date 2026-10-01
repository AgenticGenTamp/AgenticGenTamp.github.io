import sys; sys.path.insert(0,'.')
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); oc=int(sys.argv[2]); cols=sys.argv[3].split(','); dy=0.0
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':oc})
F=env.observation_space.get_type('mujoco_fixture'); M=env.observation_space.get_type('mujoco_movable_object')
fx={o.name:(obs.get(o,'x'),obs.get(o,'y')) for o in obs.get_objects(F)}
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
ap.push_to=float(sys.argv[4]) if len(sys.argv)>4 and sys.argv[4]!='none' else None
ap.slots=[(fx['cupboard_'+c.split(':')[0]][0], fx['cupboard_'+c.split(':')[0]][1]+float(c.split(':')[1]), float(c.split(':')[2])) for c in cols]
if len(sys.argv)>5: ap.rod_slot={'cuboid_%d'%i: sl for i,sl in enumerate(ap.slots)}
term=False
for t in range(int(__import__("os").environ.get("MAXT","1000"))):
    obs,r,term,tr,info=env.step(ap.get_action(obs))
    if term: break
print(sys.argv[1:], 'term', term, t+1, [np.round([obs.get(o,k) for k in 'xyz'],3).tolist() for o in obs.get_objects(M)])
