from env_client import make_env
from approach import GeneratedApproach
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor
import numpy as np
class Scan(GeneratedApproach):
 def plan_place(self,s):
  self.labels=[]
  rot=Rotation.from_euler('z',np.pi/2).as_matrix()@np.diag([1.,-1.,-1.])
  positions=[self.xyz(s,f) for f in self.fixtures]
  print('FIXTURES',self.z,[(f.name,p.tolist()) for f,p in zip(self.fixtures,positions)],flush=True)
  for fi,p in enumerate(positions):
   for x in [p[0]-.35,p[0]-.15,p[0],p[0]+.1]:
    target=[x-.11,p[1],self.z];base=[x-.66,p[1],0]
    self.add(target,rot,base,1,12);self.labels.append((fi,x,p[1],self.z))
  self.stage=2
 def plan_pick(self,s):
  if self.index: self.stage=3;return
  obj=self.objects[self.index];p=self.xyz(s,obj)
  quat=[s.get(obj,f) for f in ('qx','qy','qz','qw')]
  objrot=Rotation.from_quat(quat).as_matrix();p=p+.11*objrot[:,1]
  yaw=Rotation.from_quat(quat).as_euler('xyz')[2]
  yaw=(yaw+np.pi/2)%np.pi-np.pi/2
  rot=Rotation.from_euler('z',yaw).as_matrix()@np.diag([1.,-1.,-1.])
  base=np.array([p[0]-.55,p[1],0.]);self.lastq=None
  for z,grip,wait in [(.2,0,5),(.005,0,5),(.005,1,8),(.3,1,5)]:
   self.add([p[0],p[1],z],rot,base,grip,wait)
  self.stage=1

def trial(z):
 e=make_env();s,info=e.reset(seed=0,options={'object_count':1}); a=Scan(e.action_space,e.observation_space,{});a.z=z;a.reset(s,info);last=None
 for step in range(950):
  s,r,te,tr,_=e.step(a.get_action(s));status=(a.stage,len(a.queue));obj=a.objects[0]
  if status!=last or r!=-1 or te:
   quat=[s.get(obj,f) for f in ('qx','qy','qz','qw')]
   print('SCAN',z,step,status,'reward',r,'done',te,'obj',np.round(a.xyz(s,obj),4).tolist(),'quat',np.round(quat,3).tolist(),flush=True);last=status
  if te or tr or a.stage==3: break
 e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(trial,[.03,.15,.3,.45]))
