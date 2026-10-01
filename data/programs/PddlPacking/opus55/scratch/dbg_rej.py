import sys; sys.path.insert(0,'.')
import numpy as np
from env_client import make_env
from approach import GeneratedApproach, cfg_diff
seed=int(sys.argv[1]); upto=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
acts=[]
for i in range(upto):
    a=ap.get_action(obs); acts.append(a); obs,*_=env.step(a)
cfg=ap._parse(obs)[0]
a=ap.get_action(obs)
print("cfg",cfg.round(3)); print("act",a.round(3))
# try partial actions using fresh env replays
def test(mask):
    e=make_env(); o,_=e.reset(seed=seed)
    for b in acts: o,*_=e.step(b)
    aa=a.copy(); aa[:10]*=mask; aa[10]=0
    o2,*_=e.step(aa)
    c2=ap._parse(o2)[0]; e.close()
    return np.abs(c2-cfg).max()>1e-6
print("base only", test(np.r_[np.ones(3),np.zeros(7)]))
print("arm only", test(np.r_[np.zeros(3),np.ones(7)]))
print("full", test(np.ones(10)))
for j in range(10):
    m=np.zeros(10); m[j]=1
    print(j, test(m))
from kin import fk_full
from approach import arm_boxes
c=cfg+np.r_[a[:10]]
pts,_,_,tool,R=fk_full(c[:3],c[3:])
for nm,p in zip(["sh","up","elb","wr","tool"],pts): print(nm,p.round(3))
print("tool x axis",R[:,0].round(3))
cfgb=ap._parse(obs)
print(ap.plan[0][0].round(3) if ap.plan else None)
