from env_client import make_env
from approach import GeneratedApproach
import numpy as np
for mode in ['forward','retract','yawleft','yawright']:
 e=make_env();s,i=e.reset(seed=56);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 print(mode,'initial',s.tolist(),flush=True)
 for t in range(130):
  a=p.get_action(s);s,*_=e.step(a)
  if p.phase>0:print(mode,'normal grasp',t,flush=True);break
 print(mode,'probe_start',s[:12].tolist(),'basegoal',p.base.tolist(),flush=True)
 oldhook=s[9:12].copy()
 for t in range(80):
  u=np.array([np.cos(s[2]),np.sin(s[2])]);a=np.array([*(u*.0005),0,0,1],np.float32)
  if mode=='retract' and t<20:a[:2]=0;a[3]=-.005
  if mode=='yawleft':a[2]=.002
  if mode=='yawright':a[2]=-.002
  old=s.copy();s,*_=e.step(a)
  if t%10==0:print(mode,t,'robot',s[:5].round(5),'dh',(s[9:12]-oldhook).round(5),flush=True)
  if np.linalg.norm(s[9:12]-oldhook)>.001:print(mode,'grasped',t,flush=True);break
 e.close()
