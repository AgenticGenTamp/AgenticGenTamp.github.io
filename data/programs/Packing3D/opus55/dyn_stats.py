from env_client import make_env
import numpy as np, collections
env = make_env()
print("max_steps", getattr(env,'max_steps',None))
tf = env.observation_space.type_features
cnt=collections.Counter(); types=collections.Counter(); tt=collections.Counter()
px=[];py=[];pz=[];quats=set();sides=collections.Counter();racks=set();robots=set()
for seed in range(200):
    obs, info = env.reset(seed=seed)
    cnt[info.get('object_count')]+=1
    for name in obs.get_object_names():
        o = obs.get_object_from_name(name) if isinstance(name,str) else name
        n=str(o); t=o.type; f={k:float(obs.get(o,k)) for k in tf[t]}
        if 'robot' in n:
            robots.add(tuple(round(v,4) for v in f.values())); continue
        if 'rack' in n:
            racks.add(tuple(round(v,4) for v in f.values())); continue
        types[str(t)]+=1
        px.append(f['pose_x']);py.append(f['pose_y']);pz.append(f['pose_z'])
        q=tuple(round(f[k],3) for k in ['pose_qx','pose_qy','pose_qz','pose_qw'])
        quats.add(q)
        if 'triangle_type' in f: tt[f['triangle_type']]+=1; sides[(round(f['side_a'],4),round(f['side_b'],4),round(f['depth'],4))]+=1
        else: sides[('cub',round(f['half_extent_x'],4),round(f['half_extent_y'],4),round(f['half_extent_z'],4))]+=1
print("object_count",cnt); print("types",types); print("tri_type",tt)
print("x",min(px),max(px),"y",min(py),max(py),"z",min(pz),max(pz))
print("quats",quats); print("sizes",sides)
print("racks",racks); print("robots n",len(robots), list(robots)[:1])
print("rack feats", tf[[o for o in obs.get_object_names()][0].type] if False else '')
