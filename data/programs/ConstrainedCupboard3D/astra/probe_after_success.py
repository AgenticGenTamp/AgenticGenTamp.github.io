from env_client import make_env
from approach import GeneratedApproach
import numpy as np
e=make_env();s,info=e.reset(seed=0,options={'object_count':1})
a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info)
last=None;was=False;after=0
for t in range(800):
 try:s,r,te,tr,info=e.step(a.get_action(s))
 except Exception as exc:print('ERROR',t,str(exc),flush=True);break
 key=(a.stage,len(a.queue));pos=a.xyz(s,a.objects[0]);quat=[s.get(a.objects[0],f) for f in ['qw','qx','qy','qz']]
 if key!=last or te!=was or (a.stage==3 and after%10==0):
  print('STEP',t,'key',key,'pos',np.round(pos,5),'quat',np.round(quat,4),'term',te,'trunc',tr,'reward',r,flush=True)
 last=key;was=te
 if a.stage==3:
  after+=1
  if after>=40:break
e.close()
