import math
import numpy as np

class HookGrasp:
 def __init__(self, action_space, observation_space):
  self.observation_space=observation_space
 def reset(self,state):
  self.phase=0;self.t=0;self.goal=None;self.angle=None;self.timer=0;self.attempt=0
 def objects(self,state):
  return state.get_objects(self.observation_space.get_type('kin_robot'))[0],state.get_objects(self.observation_space.get_type('hook'))[0]
 def get_action(self,state):
  r,h=self.objects(state);self.t+=1;self.timer+=1
  if state.get(h,'held'):return np.zeros(5,dtype=np.float32)
  x,y,theta=[state.get(r,f) for f in ['x','y','theta']]
  hx,hy,ha=[state.get(h,f) for f in ['x','y','theta']]
  u=np.array([math.cos(ha),math.sin(ha)]);v=np.array([-u[1],u[0]])
  tip=np.array([hx,hy])-state.get(h,'length_side1')*u-state.get(h,'width')*.5*v
  if self.phase==0:
   self.angle=ha
   self.goal=tip-.50*u
   self.goal=np.clip(self.goal,[.25,.25],[3.24,1.46])
   dest=np.array([x,.28]);targetangle=ha;grip=.019
   if abs(y-.28)<.02 and abs(theta-ha)<.02:self.phase=1;self.timer=0
  elif self.phase==1:
   dest=np.array([self.goal[0],.28]);targetangle=self.angle;grip=.019
   if abs(x-dest[0])<.02:self.phase=2;self.timer=0
  elif self.phase==2:
   dest=self.goal;targetangle=self.angle;grip=.019
   if np.linalg.norm(dest-np.array([x,y]))<.02:self.phase=3;self.timer=0
  elif self.phase==3:
   dest=tip-.28*u;dest=np.clip(dest,[.25,.25],[3.24,1.46]);targetangle=ha;grip=.019
   if np.linalg.norm(dest-np.array([x,y]))<.012 and abs(theta-ha)<.02:self.phase=4;self.timer=0
   if self.timer>100:self.phase=4;self.timer=0
  else:
   dest=np.array([x,y]);targetangle=theta;grip=-.019
   if self.timer>15:
    self.phase=0;self.timer=0;self.attempt+=1
  delta=dest-np.array([x,y]);dt=(targetangle-theta+math.pi)%(2*math.pi)-math.pi
  return np.array([*np.clip(delta,-.049,.049),np.clip(dt,-.064,.064),-.099,grip],dtype=np.float32)
