import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); rng=np.random.default_rng(0)
res=[];fails=[]
starts=[(2.0,-3.0),(2.5,-2.5),(-2.0,-2.2),(-3.0,-2.2),(1.6,-3.5),(2.0,2.0),(-3.0,4.0),(0.0,5.0),(2.6,-3.8)]
for s in range(120):
    o,i=env.reset(seed=s)
    sx,sy=starts[s%len(starts)]
    ok=True
    for k in range(80):
        a=np.zeros(11); a[0]=np.clip(sx-o[0],-0.4,0.4); a[1]=np.clip(sy-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o2,r,t,tr,_=env.step(a)
        if np.allclose(o2[:2],o[:2]): break
        o=o2
    start=o[:2].copy()
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(o,i)
    t=False
    for k in range(400):
        o,r,t,tr,_=env.step(ap.get_action(o))
        if t or tr: break
    lb=int(np.ceil(max(abs(o[19]-start[0]),abs(o[20]-start[1]))/0.4))
    res.append((k+1,lb))
    if not t: fails.append((s,tuple(np.round(start,2)),tuple(np.round(o[19:21],3))))
res=np.array(res)
print("mean",res[:,0].mean(),"lb",res[:,1].mean(),"fails",len(fails))
for f in fails[:15]:print(" ",f)
