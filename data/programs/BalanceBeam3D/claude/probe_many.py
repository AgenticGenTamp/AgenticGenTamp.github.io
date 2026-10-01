import numpy as np
from env_client import make_env
env=make_env(); R=[]
for s in range(100):
    obs,_=env.reset(seed=s); R.append(np.asarray(obs,float))
R=np.array(R)
def stat(n,sl):
    x=R[:,sl]; print(n,"min",np.round(x.min(0),4),"max",np.round(x.max(0),4),"mean",np.round(x.mean(0),4))
stat("lb xy z",slice(0,3)); stat("sb1 xy z",slice(54,57)); stat("sb2 xy z",slice(70,73))
stat("ss xyz",slice(38,41)); stat("ss quat",slice(41,45)); stat("ss bbox",slice(51,54))
stat("base",slice(16,19))
# pairwise min distance among blocks
P=np.stack([R[:,0:2],R[:,54:56],R[:,70:72]],1)
d=[np.linalg.norm(P[:,i]-P[:,j],axis=1) for i,j in [(0,1),(0,2),(1,2)]]
print("min interblock dist", np.round(min(x.min() for x in d),4))
print("ss quat uniq rows", np.unique(np.round(R[:,41:45],4),axis=0))
# first-step reward each seed with zero action
a=np.zeros(11,dtype=np.float32); rs=[]
for s in range(20):
    env.reset(seed=s); o,r,te,tr,_=env.step(a); rs.append(r)
print("first-step rewards uniq", np.unique(rs))
