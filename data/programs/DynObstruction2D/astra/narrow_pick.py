import math
import numpy as np
class NarrowPick:
 def __init__(self,space):self.space=space
 def reset(self,state):
  self.phase='rise';self.count=0;self.xoff=0;self.yoff=.5;self.goal=None
  r=next(iter(state.get_objects(self.space.get_type('kin_robot'))))
  self.theta=state.get(r,'theta')
 def get_action(self,state):
  r=next(iter(state.get_objects(self.space.get_type('kin_robot'))));o=next(iter(state.get_objects(self.space.get_type('target_block'))));s=next(iter(state.get_objects(self.space.get_type('target_surface'))))
  g=lambda o,f:state.get(o,f)
  rx,ry,rt=g(r,'x'),g(r,'y'),g(r,'theta');ox,oy=g(o,'x'),g(o,'y')
  self.count+=1
  x,y,t,gap=rx,1.3,self.theta,.32
  if self.phase=='rise':
   if ry>1.27:self.phase='above'
  elif self.phase=='above':
   x=ox;t=-math.pi/2
   if abs(rx-x)<.008 and abs(rt-t)<.01:
    self.phase='descend';self.goal=(ox,oy+g(o,'height')/2+.28)
  elif self.phase=='descend':
   x,y=self.goal;t=-math.pi/2
   if abs(rx-x)<.008 and abs(ry-y)<.008:self.phase='close';self.count=0
  elif self.phase=='close':
   x,y=self.goal;t=-math.pi/2;gap=0
   if g(o,'held'):
    self.phase='drag';self.xoff=rx-ox;self.yoff=ry-oy
   elif self.count>25:
    self.goal=(ox,y-.015);self.count=0
  elif self.phase=='drag':
   t=rt-math.atan2(math.sin(g(o,'theta')),math.cos(g(o,'theta')));gap=0
   x=rx+np.clip(g(s,'x')-ox,-.035,.035)
   y=ry+.1+g(o,'height')/2+.008-oy
   if abs(g(s,'x')-ox)<.005:self.phase='release';self.count=0
  elif self.phase=='release':
   t=-math.pi/2;y=ry;gap=.32
   if self.count>10:self.phase='rise'
  return [x-rx,y-ry,t-rt,.24-g(r,'arm_joint'),gap-g(r,'finger_gap')]
