import math
import numpy as np
from env_client import make_env
from baseline_grasp import GeneratedApproach as Base
class Pull(Base):
 def reset(self,s,i):
  super().reset(s,i); self.stage=0;self.count=0
 def get_action(self,s):
  h=s.get_objects(self.ht)[0];r=s.get_objects(self.rt)[0];t=s.get_objects(self.tt)[0]
  if not s.get(h,'held'):return super().get_action(s)
  rx,ry,rt=[s.get(r,f) for f in ('x','y','theta')]
  hx,hy,ht=[s.get(h,f) for f in ('x','y','theta')]
  tx,ty=[s.get(t,f) for f in ('x','y')]
  if self.stage==0:self.initialx=rx;self.tx=tx
  targets=[(self.initialx,.25,math.pi/2),(.25,.25,math.pi/2),(.25,.25,2.1),(.25,1.48,2.1),(.25,1.48,math.pi/2),(.25,.25,math.pi/2)]
  x,y,a=targets[min(self.stage,len(targets)-1)]
  if self.stage>=5:x=max(.25,tx-.13);y=max(.25,ry-.035)
  theta=rt+(a-ht+math.pi)%(2*math.pi)-math.pi
  if abs(rx-x)<.025 and abs(ry-y)<.025 and abs((a-ht+math.pi)%(2*math.pi)-math.pi)<.025:
   self.stage+=1
   if self.stage<7:print('stage',self.stage,'target',tx,ty,'hook',hx,hy,ht,flush=True)
  return self.move(s,r,x,y,theta,gap=.12)
if __name__=='__main__':
 for seed in [0]:
  e=make_env();s,i=e.reset(seed=seed);p=Pull(e.action_space,e.observation_space,{});p.reset(s,i)
  for k in range(500):
   s,r,d,tr,i=e.step(p.get_action(s))
   if k%20==0:
    h=s.get_objects(p.ht)[0];t=s.get_objects(p.tt)[0];rob=s.get_objects(p.rt)[0]
    print(k,'target',[round(s.get(t,f),3) for f in ('x','y')],'hook',[round(s.get(h,f),3) for f in ('x','y','theta','held')],'robot',[round(s.get(rob,f),3) for f in ('x','y')],flush=True)
   if d or tr:break
  print('RESULT',seed,k,d,tr,flush=True);e.close()
