from env_client import make_env
from approach import GeneratedApproach
from kin import planar_ik,fk
import numpy as np,sys
class Horizontal(GeneratedApproach):
 def plan_place(self,s):
  fixture=self.fixtures[int(sys.argv[2]) if len(sys.argv)>2 else 0];obj=self.objects[self.index]
  fp,fr=fk(self.joints(s));base0=np.array([s.get(self.robot,'pos_base_x'),s.get(self.robot,'pos_base_y'),.4])
  held=fr.T@(self.xyz(s,obj)-base0-np.array([.12,0,0])-fp)
  target=self.xyz(s,fixture)+np.array([-.02,0,float(sys.argv[1]) if len(sys.argv)>1 else .20])
  rotation=np.array([[0,1.,0],[1.,0,0],[0,0,-1.]])
  tool=target-rotation@held
  print('HELD',held,'TOOL',tool,flush=True)
  for pos in [tool+[-.4,0,0],tool]:
   q,err=planar_ik(.4,pos[2]-.4,np.pi/2)
   base=np.array([pos[0]-.52,pos[1]-.00135,0.])
   self.queue.append((q,base,1,6))
  self.queue.append((q,base.copy(),0,10));self.queue.append((q,base+[-.4,0,0],0,6));self.stage=2
 def get_action(self,s):
  if self.stage==2 and self.waysteps>65:self.waysteps=181
  a=super().get_action(s)
  if self.stage==2:a[:2]=np.clip(a[:2],-.025,.025)
  return a

e=make_env();s,info=e.reset(seed=0,options={'object_count':1});a=Horizontal(e.action_space,e.observation_space,{});a.reset(s,info);last=None
for step in range(550):
 s,r,te,tr,info=e.step(a.get_action(s));status=(a.stage,len(a.queue))
 if status!=last or te:
  obj=s.get_object_from_name('cuboid_0');print(step,status,r,te,'pose',[round(s.get(obj,f),4) for f in ['x','y','z','qw','qx','qy','qz']],flush=True);last=status
 if te or tr or(a.stage==3):break
print(e.render_state(state=s,label='horiz_final'),flush=True)
e.close()
