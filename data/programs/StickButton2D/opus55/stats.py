from env_client import make_env
import numpy as np
env=make_env()
S=[];B=[];R=[];N=[]
for seed in range(200):
    obs,info=env.reset(seed=seed)
    s=obs.get_object_from_name('stick'); r=obs.get_object_from_name('robot')
    S.append([obs.get(s,f) for f in ['x','y','theta']]); R.append([obs.get(r,f) for f in ['x','y','theta']])
    bs=[o for o in obs.get_objects(env.observation_space.get_type('circle'))]
    N.append(len(bs))
    for b in bs: B.append([obs.get(b,'x'),obs.get(b,'y')])
S,B,R=map(np.array,(S,B,R))
print('stick min',S.min(0),'max',S.max(0))
print('robot min',R.min(0),'max',R.max(0))
print('btn min',B.min(0),'max',B.max(0))
print('counts',np.bincount(N))
print('btn y hist',np.histogram(B[:,1],bins=[0,0.3,0.6,0.9,1.2,1.25,1.3,1.5,2,2.3,2.4,2.5,3]))
