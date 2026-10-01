import sys; sys.path.insert(0,'/sandbox')
from env_client import make_env
env=make_env()
for seed,k in [(0,5),(3,None),(2,None),(4,None)]:
    obs,info=env.reset(seed=seed, options={} if k is None else {'object_count':k})
    F=env.observation_space.get_type('mujoco_fixture')
    print(seed,k,info,sorted([(round(obs.get(o,'y'),2),o.name) for o in obs.get_objects(F)]))
