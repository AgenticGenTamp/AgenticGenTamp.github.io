import numpy as np,json,time
from env_client import make_env
from approach import GeneratedApproach
class GoalProbe(GeneratedApproach):
 def get_action(self,s):
  if self.phase==5:
   self.goal=np.asarray(s[:3]).copy()
   self.goal[:2]+=np.array([.12 if self.item==0 else -.12,-.105 if self.item==0 else .105])
  return super().get_action(s)
e=make_env();s,i=e.reset(seed=0);p=GoalProbe(e.action_space,e.observation_space,{});p.reset(s,i);old=-1;st=time.time()
for k in range(900):
 a=p.get_action(s);s,r,t,tr,i=e.step(a)
 if p.phase!=old or t:
  print(k,'phase',p.phase,'item',p.item,'r',r,'t',t,'obj',np.round(s[[0,1,2,16,17,18,32,33,34]],5).tolist(),'info',i,flush=True);old=p.phase
 if t or tr:break
print('elapsed',time.time()-st,flush=True)
json.dump(s.tolist(),open('goal_x_state.json','w'));e.close()
