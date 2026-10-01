from env_client import make_env
import numpy as np, collections
env=make_env(); T=env.observation_space.get_type
f=env.observation_space.type_features[T("robot")]
print(list(f))
cnt=collections.Counter(); xs=[];ys=[];yaws=[];zs=[]; r0=None; same=True
for s in range(301):
    o,info=env.reset(seed=s)
    cnt[info.get('object_count')]+=1
    r=o.get_objects(T("robot"))[0]; rs=np.array([o.get(r,x) for x in f])
    if r0 is None: r0=rs; print("robot0",rs.round(4))
    elif not np.allclose(rs,r0): same=False
    for b in o.get_objects(T("block")):
        xs.append(o.get(b,"pose_x"));ys.append(o.get(b,"pose_y"));zs.append(o.get(b,"pose_z"))
        yaws.append(2*np.arctan2(o.get(b,"pose_qz"),o.get(b,"pose_qw")))
print(cnt, "robot same",same)
for n,a in [("x",xs),("y",ys),("z",zs),("yaw",yaws)]: print(n,np.min(a),np.max(a))
print([t.name for t in o.get_objects(T("block"))[:1]], info.keys())
print(env.observation_space.type_features)
