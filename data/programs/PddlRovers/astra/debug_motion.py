from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();s,info=E.reset(seed=4);a=GeneratedApproach(E.action_space,E.observation_space,{});a.reset(s,info)
for t in range(7):s,*_=E.step(a.get_action(s))
p=a.xy(s,a.rovers[0]);key=a.goal[0];f=a.field(key);d=np.max(abs(a.nodes-p),axis=1);ids=np.where(d<=.601)[0];cost=f[ids]+.95*d[ids]+np.linalg.norm(a.nodes[ids]-p,axis=1)*.002
print(p,key,f[a.nearest(p)],a.safe(p))
for j in ids[np.argsort(cost)[:60]]:
 q=a.nodes[j];ok=a.safe(np.array([p+(q-p)*z for z in [.2,.4,.6,.8,1.]])).all()
 print(q.round(2),round(f[j],4),round(f[j]+.95*d[j],4),ok)
E.close()
