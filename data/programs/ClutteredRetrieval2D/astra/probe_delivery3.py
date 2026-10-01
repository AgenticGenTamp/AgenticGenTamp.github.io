from env_client import make_env
from approach import GeneratedApproach
from geometry import Scene
import numpy as np,math
class Probe(GeneratedApproach):
 def carry_goals(self,idx):
  goals=super().carry_goals(idx)
  print('CARRY',idx,self.chosen,'region',self.region,'held',self.held,'q',self.q,'ngoals',len(goals),flush=True)
  if self.chosen=='target_block':
   print('rects',[(o.name,self.rects[j].tolist()) for j,o in enumerate(self.objects)],flush=True)
   for a in [self.q[2]]+list(np.linspace(-math.pi,math.pi,16,endpoint=False)):
    u=np.array([math.cos(a),math.sin(a)]);v=np.array([-u[1],u[0]])
    q=np.r_[self.region[:2]-u*self.held[0]-v*self.held[1],a,.2]
    fail=[]
    for j,o in enumerate(self.objects):
     if j!=idx and not Scene(self.rects[j:j+1]).valid(q,held=self.held):fail.append(o.name)
    print('candidate',q,'fails',fail,'walls',Scene(np.empty((0,5))).valid(q,held=self.held),flush=True)
   raise StopIteration
  return goals
e=make_env();s,i=e.reset(seed=3);p=Probe(e.action_space,e.observation_space,{});p.reset(s,i)
try:
 for k in range(300):
  a=p.get_action(s);s,r,d,tr,i=e.step(a)
  if d or tr:break
except StopIteration:pass
e.close()
