from env_client import make_env
from approach import GeneratedApproach
from kin import planar_ik,fk
import numpy as np,sys,time
class Vertical(GeneratedApproach):
 def plan_place(self,s):
  p=self.xyz(s,self.fixtures[0]);z=float(sys.argv[1]) if len(sys.argv)>1 else .2
  for x in [p[0]-.5,p[0]-.25,p[0]-.1,p[0],p[0]+.06]:
   q,err=planar_ik(.65,z-.4,np.pi/2,pitch=np.pi/2)
   print('PLAN',x,z,'err',err,flush=True)
   self.queue.append((q,np.array([x-.8,p[1],0]),1,8))
  self.stage=3

e=make_env();s,info=e.reset(seed=0,options={'object_count':1});a=Vertical(e.action_space,e.observation_space,{});a.reset(s,info);last=None
for step in range(500):
 s,r,te,tr,info=e.step(a.get_action(s));status=(a.stage,len(a.queue))
 if status!=last:
  obj=s.get_object_from_name('cuboid_0');print(step,status,r,te,'pose',[round(s.get(obj,f),4) for f in ['x','y','z','qw','qx','qy','qz']], 'q',a.joints(s).round(3),flush=True)
  print(e.render_state(state=s,label='vertical_'+str(step)),flush=True);last=status
 if te or tr or (a.stage==3 and not a.queue):break
e.close()
