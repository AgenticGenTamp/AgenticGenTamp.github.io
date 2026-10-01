from env_client import make_env
from grasp_policy import GeneratedApproach
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def trial(pos,steps=1):
 vv=[0,12,12]
 e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);p.yaw=0;p.q=p.ik(.55,.10);rob=p.robot;c=p.cube;b=s.get_object_from_name('bin_0')
 def qnow():return np.array([s.get(rob,'pos_arm_joint'+str(j)) for j in range(1,8)])
 for k in range(125):s,*_=e.step(p.get_action(s))
 p.yaw=0;qt=p.ik(*pos);base=np.array([.6,s.get(b,'y')-.00135,0])
 for k in range(55):
  a=np.zeros(18);err=qt-qnow();a[3:10]=np.clip(err,-.1,.1);a[11:]=5*err;a[10]=1;a[:3]=np.clip(base-np.array([s.get(rob,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]),-.1,.1);s,*_=e.step(a)
 vals=[];first_done=None;predictions=None
 initial=[round(s.get(c,f),6) for f in ['x','y','z','vx','vy','vz']]
 for k in range(18):
  a=np.zeros(18)
  if k<=steps:a[[12,14,16]]=vv
  a[10]=float(k<steps)
  s,r,t,tr,i=e.step(a)
  if t and first_done is None:first_done=(k,[round(s.get(c,f),6) for f in ['x','y','z','vx','vy','vz']])
  if k==1:
   px,pz,vx,vz=[s.get(c,f) for f in ['x','z','vx','vz']];predictions={z:round(px+vx*(vz+np.sqrt(vz*vz+19.62*(pz-z)))/9.81,6) for z in [.05,.225]}
  if k<=steps+2 or k==17:vals.append((k,[round(s.get(c,f),4) for f in ['x','y','z','vx','vy','vz']],r,t))
 e.close();return {'pos':pos,'vv':vv,'initial':initial,'vals':vals,'first_done':first_done,'landing_prediction':predictions}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as ex:
  for out in ex.map(lambda v:trial(v),[(x,z) for x in [.35,.4,.45] for z in [.6,.65,.7]]):print(out,flush=True)
