from approach import GeneratedApproach
from env_client import make_env
from kin import fk,planar_ik
from concurrent.futures import ThreadPoolExecutor
import numpy as np,time
class Low(GeneratedApproach):
 def plan_place(self,s):
  fixture=self.fixtures[0];obj=self.objects[self.index];fp,fr=fk(self.joints(s));robotbase=np.array([s.get(self.robot,'pos_base_x'),s.get(self.robot,'pos_base_y'),.4]);held=fr.T@(self.xyz(s,obj)-robotbase-[.12,0,0]-fp);self.held=held
  target=self.xyz(s,fixture)+[0,0,self.z];pitch=np.pi/4;yaw=np.pi/2;_,rotation=fk(planar_ik(.65,-.1,yaw,pitch)[0]);tool=target-rotation@held
  currentbase=np.array([s.get(self.robot,'pos_base_x'),s.get(self.robot,'pos_base_y'),0.]);q,err=planar_ik(.55,-.1,yaw,pitch);self.queue.append((q,currentbase,1,5))
  for pos in [tool+[-.4,0,0],tool]:
   q,err=planar_ik(.65,pos[2]-.4,yaw,pitch);base=np.array([pos[0]-.77,pos[1]-.00135,0.]);self.queue.append((q,base,1,5))
  self.queue.append((q,base.copy(),0,8));self.queue.append((q,base+[-.4,0,0],0,5));self.stage=2

def trial(z):
 e=make_env();s,info=e.reset(seed=0,options={'object_count':1});a=Low(e.action_space,e.observation_space,{});a.z=z;a.reset(s,info);last=None
 for step in range(650):
  s,r,te,tr,_=e.step(a.get_action(s));status=(a.stage,len(a.queue));obj=a.objects[0]
  if status!=last or te:print('LOW',z,step,status,'pos',np.round(a.xyz(s,obj),4).tolist(),'term',te,flush=True);last=status
  if te or tr or a.stage==3:break
 for k in range(30):
  if te or tr:break
  s,r,te,tr,_=e.step(np.zeros(11))
 print('FINAL',z,step+1,'pos',np.round(a.xyz(s,obj),4).tolist(),'term',te,flush=True);e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(trial,[.28,.32]))
