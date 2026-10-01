import numpy as np,sys,time
from env_client import make_env
from approach import GeneratedApproach
xoff=float(sys.argv[1]);e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();bowl=s[:3].copy();last=None
for k in range(950):
 if p.phase==5:
  p.goal=bowl.copy();p.goal[0]+=xoff;p.goal[1]+=-.005 if p.item==0 else .005
 old=(p.phase,p.item);s,r,t,tr,i=e.step(p.get_action(s))
 if old!=last or t:print(k,old,'r',r,'t',t,'objs',np.round(s[[0,1,2,16,17,18,32,33,34]],4).tolist(),flush=True)
 last=old
 if t or tr:break
print('END',k,r,t,tr,'time',time.time()-start,flush=True);e.close()
