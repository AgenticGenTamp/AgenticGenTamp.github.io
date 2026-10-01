from env_client import make_env
import numpy as np
env = make_env()
P={'target_region':[], 'target_block':[], 'obs':[]}; cnt=[]
for s in range(150):
    o,info=env.reset(seed=s); cnt.append(info['object_count'])
    for n in o.get_object_names():
        if n=='robot': continue
        ob=o.get_object_from_name(n); v=[o.get(ob,f) for f in ('pose_x','pose_y','pose_z','half_extent_x','half_extent_y','half_extent_z')]
        P['obs' if n.startswith('obst') else n].append(v)
for k,v in P.items():
    v=np.array(v); print(k, 'min',v.min(0).round(3),'max',v.max(0).round(3))
print('counts', np.bincount(cnt))
