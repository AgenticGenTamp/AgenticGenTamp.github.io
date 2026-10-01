from probe_single import trial
from approach import GeneratedApproach
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import probe_single
class Centered(GeneratedApproach):
 def plan_pick(self,s):
  if self.index>=len(self.objects):self.stage=3;return
  obj=self.objects[self.index];p=self.xyz(s,obj);quat=[s.get(obj,f) for f in ('qx','qy','qz','qw')];yaw=Rotation.from_quat(quat).as_euler('xyz')[2];yaw=(yaw+np.pi/2)%np.pi-np.pi/2
  p=p+np.array([-np.sin(yaw),np.cos(yaw),0.])*self.offset;rot=Rotation.from_euler('z',yaw).as_matrix()@np.diag([1.,-1.,-1.]);base=np.array([p[0]-.52,p[1],0.]);self.lastq=None
  for z,grip,wait in [(.2,0,5),(.005,0,5),(.005,1,8),(.3,1,5)]:self.add([p[0],p[1],z],rot,base,grip,wait)
  self.stage=1
if __name__=='__main__':
 import sys
 Centered.offset=float(sys.argv[1]);probe_single.GeneratedApproach=Centered
 from env_client import make_env
 from kin import fk
 import time
 seed=int(sys.argv[2]) if len(sys.argv)>2 else 5
 count=int(sys.argv[3]) if len(sys.argv)>3 else 1
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':count} if count else None);a=Centered(e.action_space,e.observation_space,{});a.reset(s,info);start=time.time();picked=set()
 for step in range(950):
  s,r,te,tr,_=e.step(a.get_action(s))
  if a.stage==1 and not a.queue and a.index not in picked:
   print('PICK',Centered.offset,seed,count,a.index,np.round(a.xyz(s,a.objects[a.index]),4).tolist(),flush=True);picked.add(a.index)
  if te or tr or (a.stage==3 and not a.queue):break
 print('OFFSET',Centered.offset,seed,count,'steps',step+1,'term',te,'trunc',tr,'elapsed',round(time.time()-start,2),'objects',[(o.name,np.round(a.xyz(s,o),4).tolist()) for o in a.objects],flush=True)
 e.close()
