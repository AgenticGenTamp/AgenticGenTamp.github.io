from env_client import make_env
from approach import GeneratedApproach
import numpy as np,sys
close=float(sys.argv[1]) if len(sys.argv)>1 else 1
for m in [.30,.33,.36,.39]:
 for dx in [-.11,-.12,-.13]:
  e=make_env();s,i=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.mount=m;p.close=close;p.reset(s,i);p.b[0]+=dx;mx=0;phase=-1
  for k in range(320):
   s,*_=e.step(p.get_action(s));mx=max(mx,s[2])
   if p.phase==3 and p.age>65:break
  print('FINE',close,m,dx,'maxz',round(mx,4),'last',np.round(s[:3],4),flush=True);e.close()
