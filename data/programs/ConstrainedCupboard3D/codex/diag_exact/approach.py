import numpy as np
class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):self.a=action_space;self.o=observation_space;self.q=np.array([-.28471,.62423,1.27544,-.83415,.35352,-.04872,-1.33449])
 def reset(self,s,info):
  self.k=0;t=self.o.get_type('mujoco_movable_object');o=sorted(s.get_objects(t),key=lambda x:float(s.get(x,'x')))[0];self.xy=[float(s.get(o,'x')),float(s.get(o,'y'))]
 def get_action(self,s):
  self.k+=1;a=np.zeros(self.a.shape,dtype=self.a.dtype);a[-1]=1;r=s.get_object_from_name('robot');a[0]=np.clip((self.xy[0]-.68-float(s.get(r,'pos_base_x')))/.87,-.1,.1);a[1]=np.clip((self.xy[1]-.11-float(s.get(r,'pos_base_y')))/.87,-.1,.1);x=np.array([float(s.get(r,f'pos_arm_joint{i}')) for i in range(1,8)]);e=(self.q-x+np.pi)%(2*np.pi)-np.pi;a[3:10]=np.clip(.5*e,-.1,.1);return a
