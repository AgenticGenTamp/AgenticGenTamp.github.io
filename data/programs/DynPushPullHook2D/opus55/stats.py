import numpy as np
from env_client import make_env
env=make_env(); rows=[]
for seed in range(150):
    obs,info=env.reset(seed=seed)
    g=lambda n,f: float(obs.get(obs.get_object_from_name(n),f))
    obs_names=[o for o in obs.get_objects(env.observation_space.get_type('dyn_rectangle'))]
    no=info.get('object_count')
    rows.append([g('target_block','x'),g('target_block','y'),g('target_block','width'),g('hook','x'),g('hook','y'),g('hook','theta'),g('robot','x'),g('robot','y'),no])
a=np.array(rows)
for i,nm in enumerate(['tx','ty','tw','hx','hy','hth','rx','ry','nobj']):
    print(nm, a[:,i].min().round(3), a[:,i].max().round(3), a[:,i].mean().round(3))
print(np.bincount(a[:,8].astype(int)))
print('tx<1.0:', (a[:,0]<1.0).sum(), 'tx>2.5', (a[:,0]>2.5).sum())
