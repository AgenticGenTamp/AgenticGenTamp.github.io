import numpy as np, approach as A
from env_client import make_env
A._MAP={}   # pretend we know nothing about the world
env=make_env(); fails=[];steps=[]
starts=[(0,0),(2.0,-3.0),(2.5,-2.5),(-2.0,-2.2),(1.6,-3.5),(0.0,5.0),(2.6,-3.8)]
for s in range(140):
    o,i=env.reset(seed=s); sx,sy=starts[s%len(starts)]
    for k in range(80):
        a=np.zeros(11); a[0]=np.clip(sx-o[0],-0.4,0.4); a[1]=np.clip(sy-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o2,r,t,tr,_=env.step(a)
        if np.allclose(o2[:2],o[:2]): break
        o=o2
    ap=A.GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(o,i)
    t=False
    for k in range(1000):
        o,r,t,tr,_=env.step(ap.get_action(o))
        if t or tr: break
    steps.append(k+1)
    if not t: fails.append((s,(round(float(sx),1),round(float(sy),1)),tuple(np.round(o[19:21],3))))
print("mean",np.mean(steps),"fails",len(fails)); [print(" ",f) for f in fails[:10]]
