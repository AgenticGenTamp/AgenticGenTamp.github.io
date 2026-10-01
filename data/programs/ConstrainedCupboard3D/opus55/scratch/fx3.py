import sys; sys.path.insert(0,'/sandbox')
from env_client import make_env
env=make_env()
for seed,k in [(2,None),(5,6),(7,1),(8,4),(9,2),(10,3)]:
    obs,info=env.reset(seed=seed, options={} if k is None else {'object_count':k})
    F=env.observation_space.get_type('mujoco_fixture')
    print(seed,k,info['object_count'],' '.join('%s:%.2f'%(n,y) for y,n in sorted([(round(obs.get(o,'y'),2),o.name) for o in obs.get_objects(F)])))
