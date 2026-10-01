from env_client import make_env
import numpy as np
env = make_env()
gaps=[]; counts=[]; rs=[]; ts=[]
for seed in range(300):
    obs, info = env.reset(seed=seed)
    counts.append(info['object_count'])
    obst = obs.get_objects([t for t in env.observation_space.types if t.name=='rectangle'][0])
    walls={}
    for o in obst:
        d={f:obs.get(o,f) for f in ['x','y','width','height','theta']}
        if o.name=='target_region': continue
        walls.setdefault(round(d['x'],3),[]).append((d['y'],d['y']+d['height'],d['theta'],d['width']))
    for x,segs in walls.items():
        segs.sort()
        for a,b in zip(segs,segs[1:]): gaps.append(b[0]-a[1])
        if any(s[2]!=0 for s in segs): print('rot',seed)
    r=obs.get_object_from_name('robot'); t=obs.get_object_from_name('target_region')
    rs.append((obs.get(r,'x'),obs.get(r,'y'))); ts.append((obs.get(t,'x'),obs.get(t,'y')))
print('counts',sorted(set(counts)), np.bincount(counts))
print('gaps min/max',min(gaps),max(gaps))
rs=np.array(rs);ts=np.array(ts)
print('robot range',rs.min(0),rs.max(0)); print('target range',ts.min(0),ts.max(0))
env.close()
