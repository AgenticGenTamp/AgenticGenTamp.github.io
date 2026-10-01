import math
import numpy as np

class HookPull:
 def __init__(self, action_space, observation_space):
  self.space=observation_space
  self.rt=self.space.get_type('kin_robot');self.ht=self.space.get_type('hook');self.tt=self.space.get_type('target_block')
 def reset(self,state):
  self.stage=0;self.timer=0;self.cycle=0;self.gx=None
 def get_action(self,state):
  r=state.get_objects(self.rt)[0];h=state.get_objects(self.ht)[0];t=state.get_objects(self.tt)[0]
  rx,ry,rt=[state.get(r,f) for f in ('x','y','theta')]
  hx,hy,ht=[state.get(h,f) for f in ('x','y','theta')]
  tx,ty,tt=[state.get(t,f) for f in ('x','y','theta')]
  if self.gx is None:
   da=math.pi/2-ht
   ox=math.cos(da)*(hx-rx)-math.sin(da)*(hy-ry)
   self.gx=max(.25,min(3.24,tx-.50-ox));self.startx=rx
  self.timer+=1
  if self.stage==0:x,y,a=self.startx,.25,math.pi/2
  elif self.stage==1:x,y,a=self.gx,.25,math.pi/2
  elif self.stage==2:x,y,a=self.gx,.25,2.1
  elif self.stage==3:x,y,a=self.gx,1.48,2.1
  elif self.stage==4:x,y,a=self.gx,1.48,math.pi/2
  else:
   x=max(.25,min(3.24,tx-.27-(hx-rx)))
   y=max(.25,ry-.035);a=math.pi/2
   if self.timer>110:
    self.stage=0;self.timer=0;self.gx=None;self.cycle+=1
  da=(a-ht+math.pi)%(2*math.pi)-math.pi
  if self.stage<5 and ((abs(rx-x)<.025 and abs(ry-y)<.025 and abs(da)<.025) or self.timer>100):
   self.stage+=1;self.timer=0
  return np.array([np.clip(x-rx,-.049,.049),np.clip(y-ry,-.049,.049),np.clip(da,-.064,.064),0.,-.019],dtype=np.float32)
