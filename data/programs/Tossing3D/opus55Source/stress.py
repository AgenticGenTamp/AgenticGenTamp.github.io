import numpy as np, sys, json, time
from common import make_env
from approach import GeneratedApproach
env=make_env([1,2])
appr=GeneratedApproach(env.action_space, env.observation_space, {})
n=int(sys.argv[1]); seeds=json.loads(sys.argv[2]); res=[]
for seed in seeds:
    rng=np.random.default_rng(seed)
    s,info=env.reset(seed=seed, options={'object_count':n})
    s=s.copy()
    pts=[]
    for j in range(n):
        o=s.get_object_from_name(f'cube_{j}')
        while True:
            x,y=rng.uniform(0.35,1.2),rng.uniform(-1.0,1.0)
            if all(np.hypot(x-a,y-b)>0.12 for a,b in pts): break
        pts.append((x,y)); yaw=rng.uniform(-np.pi,np.pi)
        s.set(o,'x',x); s.set(o,'y',y); s.set(o,'qw',np.cos(yaw/2)); s.set(o,'qz',np.sin(yaw/2)); s.set(o,'qx',0); s.set(o,'qy',0)
    env.set_state(s); s=env.get_state()
    appr.reset(s,info); t=time.time(); ok=False
    for i in range(400+400*n):
        s,r,term,_,_=env.step(appr.get_action(s))
        if term: ok=True; break
    cubes=[(round(s.get(o,'x'),2),round(s.get(o,'y'),2)) for o in [s.get_object_from_name(f'cube_{j}') for j in range(n)]]
    print(f'seed={seed} ok={ok} steps={i+1} t={time.time()-t:.1f} init={np.round(pts,2).tolist()} final={cubes} fails={appr.fails}', flush=True); res.append(ok)
print('rate', np.mean(res))
