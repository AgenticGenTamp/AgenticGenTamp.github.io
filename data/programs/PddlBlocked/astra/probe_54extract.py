from env_client import make_env
from approach import GeneratedApproach
from kinematics import fk,solve
import numpy as np
for mode in ['direct','lift02','lift05','lift10','lift20','up02','up10','baseback05','basex05','basey05']:
 e=make_env();s,i=e.reset(seed=54);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i)
 for n in range(150):
  ac=a.get_action(s);s,*_=e.step(ac)
  if s.get(a.r,'grasp_active'):break
 def p():return np.array([s.get(a.r,f) for f in a.fs])
 old=p();tool=fk(old)[0];v=old.copy()
 def move(v):
  global s
  for t in range(100):
   old=p();d=v-old;d[[2,7,9]]=(d[[2,7,9]]+np.pi)%(2*np.pi)-np.pi
   if max(abs(d))<1e-5:return True
   ac=np.zeros(11);ac[:10]=d*min(1,.2/max(abs(d)));s,*_=e.step(ac)
   if max(abs(p()-old))<1e-6:return False
  return False
 if mode=='direct':v=a.high.copy()
 elif mode.startswith('lift'):v[4]-=int(mode[-2:])*.01
 elif mode.startswith('up'):v,err=solve(tool+[0,0,int(mode[-2:])*.01],a.yaw,v[:2],v)
 elif mode=='baseback05':v[:2]-=.05*a.d[:2]
 elif mode=='basex05':v[0]-=.05
 else:v[1]+=.05
 ok=move(v);print('MODE',mode,'first',ok,'delta',(fk(p())[0]-tool).round(5),'p',p().round(4),'high',a.high.round(4),flush=True)
 if ok:
  ok=move(a.high);print('HIGH',ok,flush=True)
  if ok:
   v=a.high.copy();v[:2]+=np.array([-1,.3] if a.north else [-.65,.65]);print('CARRY',move(v),flush=True)
 e.close()
