from env_client import make_env
import numpy as np

def dump(e,s):
 for t in e.observation_space.types:
  for o in s.get_objects(t):
   print(o.name,t.name,{f:round(s.get(o,f),5) for f in e.observation_space.type_features[t]})
e=make_env();s,i=e.reset(seed=0);dump(e,s);print('INFO',i)
for j in [10,0,1,2,3,4,5,6,7,8,9]:
 a=np.zeros(11);a[j]=-.99 if j==10 else .2
 ns,r,te,tr,inf=e.step(a)
 print('ACT',j,'R',r,te,tr,inf)
 for t in e.observation_space.types:
  for o in s.get_objects(t):
   dif={f:round(ns.get(o,f)-s.get(o,f),5) for f in e.observation_space.type_features[t] if abs(ns.get(o,f)-s.get(o,f))>1e-6}
   if dif:print(o.name,dif)
 s=ns
e.close()
