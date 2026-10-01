from env_client import make_env
from approach import GeneratedApproach
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def trial(vv,steps=1):
 e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);rob=p.robot;c=p.cube;b=s.get_object_from_name('bin_0')
 def qnow():return np.array([s.get(rob,'pos_arm_joint'+str(j)) for j in range(1,8)])
 for k in range(125):s,*_=e.step(p.get_action(s))
 qt=p.ik(.4,.65);base=np.array([.9,s.get(b,'y')-.00135,0])
 for k in range(55):
  a=np.zeros(18);err=qt-qnow();a[3:10]=np.clip(err,-.1,.1);a[11:]=5*err;a[10]=1;a[:3]=np.clip(base-np.array([s.get(rob,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]),-.1,.1);s,*_=e.step(a)
 vals=[]
 for k in range(18):
  a=np.zeros(18)
  if k<=steps:a[[12,14,16]]=vv
  a[10]=float(k<steps)
  s,r,t,tr,i=e.step(a)
  if k<=steps+2 or k==17:vals.append((k,[round(s.get(c,f),4) for f in ['x','y','z','vx','vy','vz']],r,t))
 e.close();return vv,steps,vals
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as ex:
  for out in ex.map(lambda v:trial(v),[[-5,8,8],[-5,6,6],[-3,6,6],[-3,8,8],[0,8,8],[-5,10,10]]):print(out,flush=True)
