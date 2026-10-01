from env_client import make_env
import numpy as np
env=make_env(); T=env.observation_space.get_type
yaws=[];he=set();surf=[]
for s in range(301):
    o,info=env.reset(seed=s)
    for b in o.get_objects(T("block")):
        y=2*np.arctan2(o.get(b,"pose_qz"),o.get(b,"pose_qw")); yaws.append((y+np.pi)%(2*np.pi)-np.pi)
        he.add(tuple(round(o.get(b,k),4) for k in ["half_extent_x","half_extent_y","half_extent_z"]))
        assert abs(o.get(b,"pose_qx"))<1e-6
    for sf in o.get_objects(T("surface")):
        surf.append((sf.name,)+tuple(round(o.get(sf,k),4) for k in env.observation_space.type_features[T("surface")]))
yaws=np.array(yaws); print("yaw",yaws.min(),yaws.max(), np.histogram(yaws,8)[0])
print("he",he); 
import collections; print(collections.Counter(surf).most_common(6)); print(len(set(surf)))
# min separation between blocks
