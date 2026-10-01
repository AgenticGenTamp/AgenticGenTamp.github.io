import numpy as np
from scipy.optimize import least_squares
from kinematics import HOME,fk
from env_client import make_env
from concurrent.futures import ThreadPoolExecutor

def pose(x,z,tilt=0):
 def res(v):
  q=HOME.copy();q[[1,3,5]]=v;p,R=fk(q)
  return np.r_[(p-np.array([x,.001,z]))[[0,2]],R[0,2]-np.sin(tilt),R[2,2]-np.cos(tilt)]
 q=HOME.copy();v=least_squares(res,HOME[[1,3,5]],max_nfev=100,bounds=([-2.24,-2.58,-2.1],[2.24,2.58,2.1]));q[[1,3,5]]=v.x
 return q

def run(par):
 deg,lift=par;seed=42;dy=0;x=.6;tilt=np.deg2rad(deg);e=make_env();s,inf=e.reset(seed=seed,options={'object_count':1});fixture=s.get_object_from_name('cupboard_1');z=s.get(fixture,'z')+lift-.095+.04*np.cos(tilt)+.055;r=s.get_object_from_name('robot');o=s.get_object_from_name('cube1');b0=np.array([s.get(o,'x')-.625,s.get(o,'y')-.001,0])
 step=0;success=False
 def move(q,b,g,n):
  nonlocal s,step,success
  for i in range(n):
   step+=1
   cur=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)]);base=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
   a=np.r_[np.clip(b-base,-.1,.1),np.clip(2*(q-cur),-.1,.1),g].astype(np.float32)
   s,rew,t,tr,inf=e.step(a)
   if t and not success: success=True;print('SUCCESS',par,'reward',rew,'step',step,'grip',g,flush=True)
  qnow=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)])
  return {'cube':np.round([s.get(o,f) for f in ['x','y','z']],4).tolist(),'base':np.round([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']],4).tolist(),'qerr':round(float(np.linalg.norm(qnow-q)),4)}
 move(HOME,b0,0,35);down=pose(.5,-.04);move(down,b0,0,95);print(par,'grasp',move(down,b0,1,15),flush=True)
 target=pose(x,z-.055,tilt)
 print(par,'target',np.round(target,3),'FK',np.round(fk(target)[0],3),flush=True)
 print(par,'lift',move(target,b0,1,100),flush=True)
 dest=np.array([s.get(fixture,'x')-x-.125,dy-.001,0.])
 print(par,'insert',move(target,dest,1,80),flush=True)
 print(par,'drop',move(target,dest,0,30),flush=True);e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(4) as ex:list(ex.map(run,[(-30,.61),(-15,.61),(-30,.63),(-15,.63)]))
