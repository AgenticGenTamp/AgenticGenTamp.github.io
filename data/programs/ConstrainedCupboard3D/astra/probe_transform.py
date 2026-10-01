from env_client import make_env
from approach import GeneratedApproach
from kin import fk,planar_ik
from scipy.spatial.transform import Rotation
import numpy as np
class Calibration(GeneratedApproach):
 def plan_place(self,s):
  self.labels=[];base=np.array([s.get(self.robot,f) for f in ('pos_base_x','pos_base_y','pos_base_rot')]);self.stage=2
  for pitch in [0,-.35,.35]:
   for yaw in [-1.2,-.4,.4,1.2]:
    q,err=planar_ik(.4,-.05,yaw=yaw,pitch=pitch,q0=self.lastq)
    self.lastq=q;self.queue.append((q,base.copy(),1,14));self.labels.append((yaw,pitch))
 def plan_pick(self,s):
  if self.index:self.stage=3;return
  super().plan_pick(s)
e=make_env();s,info=e.reset(seed=0,options={'object_count':1});a=Calibration(e.action_space,e.observation_space,{});a.reset(s,info);data=[]
for step in range(1000):
 oldstage=a.stage;oldlen=len(a.queue)
 s,r,te,tr,_=e.step(a.get_action(s))
 if a.stage==2 and len(a.queue)<oldlen:
  p,R=fk(a.joints(s));base=np.array([s.get(a.robot,f) for f in ('pos_base_x','pos_base_y','pos_base_rot')]);RB=Rotation.from_euler('z',base[2]).as_matrix();obj=a.objects[0];op=a.xyz(s,obj);quat=[s.get(obj,f) for f in ('qx','qy','qz','qw')]
  y=RB.T@(op-[base[0],base[1],0])-p;data.append((y,R));label=a.labels[len(a.labels)-len(a.queue)-1]
  print('POSE',step,label,'obj',np.round(op,5).tolist(),'objquat',np.round(quat,5).tolist(),'base',np.round(base,5).tolist(),'fk',np.round(p,5).tolist(),'R',np.round(R,5).tolist(),'residual',np.round(y,5).tolist(),flush=True)
 if a.stage==3 or te or tr:break
A=np.vstack([np.concatenate([np.eye(3),R],axis=1) for y,R in data]);b=np.concatenate([y for y,R in data]);x=np.linalg.lstsq(A,b,rcond=None)[0];res=(A@x-b).reshape(-1,3)
print('FIT mount',x[:3].tolist(),'held',x[3:].tolist(),'RMS',np.sqrt(np.mean(res**2)),'max',np.max(np.abs(res)),flush=True)
print('RESIDUALS',np.round(res,5).tolist(),flush=True)
e.close()
