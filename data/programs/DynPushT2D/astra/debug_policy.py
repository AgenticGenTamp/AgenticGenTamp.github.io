from env_client import make_env
from approach import *
import sys
seed=int(sys.argv[1]);e=make_env();s,i=e.reset(seed=seed);a=GeneratedApproach(None,None,{});a.reset(s,i)
for k in range(400):
 u=a.get_action(s)
 if k%20==0:print(k,a.mode,a.contact,a.age,'u',np.round(u,3),'err',np.round(s[29:31]-s[:2],3),round(wrap(s[31]-s[2]),2),'q',np.round(a.q[a.contact],3),'p_local',np.round((s[16:18]-s[:2])@rot(s[2]),3),'score',round(a.scores(s)[0][a.contact],3),flush=True)
 s,r,t,tr,i=e.step(u)
e.close()
