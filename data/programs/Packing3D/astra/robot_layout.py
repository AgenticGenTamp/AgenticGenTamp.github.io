from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import sys
layout=sys.argv[1] if len(sys.argv)>1 else 'rows'
N=int(sys.argv[2]) if len(sys.argv)>2 else 3
class Layout(GeneratedApproach):
 def get_action(self,s):
  a=super().get_action(s)
  if self.target is not None and self.phase in ['lift','transfer','lower','release']:
   if layout=='pair':self.drop=self.xyz(s,self.rack)+np.array([0,-.06 if self.slot==0 else .06,.005])
   if layout=='rows':self.drop=self.xyz(s,self.rack)+np.array([0,[-.1,0,.1][self.slot%3],.005])
   if layout=='grid':self.drop=self.xyz(s,self.rack)+np.array([[-.05,.05][self.slot//3%2],[-.1,0,.1][self.slot%3],.005])
   if layout=='wide':self.drop=self.xyz(s,self.rack)+np.array([[-.051,.051][self.slot//3%2],[-.101,0,.101][self.slot%3],.005])
  return a
E=make_env();s,info=E.reset(seed=0,options={'object_count':N});A=Layout(E.action_space,E.observation_space,{});A.reset(s,info);prev=None;stall=0
print('GEOM',[(n,s.get_object_from_name(n).type.name,s.get(s.get_object_from_name(n),'triangle_type') if s.get_object_from_name(n).type.name.endswith('Triangle') else '') for n in s.get_object_names() if n.startswith('part')],flush=True)
for k in range(1000):
 a=A.get_action(s);ns,r,t,tr,info=E.step(a)
 status=(A.phase,A.slot)
 if status!=prev:
  print(k,status,'target',None if A.target is None else A.target.name,'pose',None if A.target is None else A.xyz(ns,A.target),flush=True);prev=status
 if A.phase_steps>25:
  print('STUCK',k,A.phase,A.slot,'drop',getattr(A,'drop',None),flush=True);break
 s=ns
 if t or tr:
  print('END',k,t,tr,flush=True);break
print('PARTS',[(n,[s.get(s.get_object_from_name(n),'pose_'+c) for c in 'xyz']) for n in s.get_object_names() if n.startswith('part')],flush=True);E.close()
