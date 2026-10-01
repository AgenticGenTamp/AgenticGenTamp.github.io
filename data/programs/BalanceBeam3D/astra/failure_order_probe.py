import numpy as np,time,sys
from env_client import make_env
from validated_slow import GeneratedApproach as Base
class Policy(Base):
 def reset(self,state,info):
  super().reset(state,info)
  if sys.argv[1]=='swap':self.objects=[0,70,54]
  self.start_object(state)
 def start_object(self,state):
  super().start_object(state)
  if self.obj!=0 and sys.argv[1]!='swap':self.b[1]+=float(sys.argv[1])
e=make_env();s,info=e.reset(seed=307);p=Policy(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();prev=None
for k in range(e.max_steps):
 key=(p.which,p.phase,p.retries)
 if key!=prev:print(k,key,'xyz',np.round(s[[0,1,2,54,55,56,70,71,72]],4),flush=True);prev=key
 s,r,t,tr,info=e.step(p.get_action(s))
 if t or tr:break
print('RESULT',sys.argv[1],k+1,t,tr,r,info,'time',time.time()-start,'xyz',np.round(s[[0,1,2,54,55,56,70,71,72]],4),'beam',np.round(s[38:45],5),flush=True)
e.close()
