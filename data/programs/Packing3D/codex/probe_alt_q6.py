"""Use wrist redundancy to reach the rack right column from a safer base x."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

def g(s,n,f):return s.get(s.get_object_from_name(n),f)
for q6off in (-.05,-.08,-.10,-.12):
 e=make_env();s,info=e.reset(seed=0,options={'object_count':3});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);term=False
 for k in range(100):
  a=p.get_action(s);s,_,term,_,_=e.step(a)
  if p.target=='part0' and g(s,'robot','grasp_active')>.5 and p.phase=='lift':break
 # High lift plus redundant wrist extension.
 target=p.PICK_JOINTS.copy();target[1]-=.30;target[5]+=q6off
 for _ in range(5):
  a=np.zeros(11,np.float32)
  for j in range(7):a[3+j]=np.clip(target[j]-g(s,'robot','joint_%d'%(j+1)),-.2,.2)
  s,*_=e.step(a)
 # Position the observed held object directly over an open right-column target.
 for _ in range(5):
  a=np.zeros(11,np.float32);a[0]=np.clip(.375-g(s,'part0','pose_x'),-.2,.2);a[1]=np.clip(-.07-g(s,'part0','pose_y'),-.2,.2)
  s,*_=e.step(a)
 for i in range(25):
  a=np.zeros(11,np.float32);a[4]=.02;s,_,term,_,_=e.step(a)
  if term or not g(s,'robot','grasp_active'):break
 if not term and g(s,'robot','grasp_active'):
  for _ in range(5):
   a=np.zeros(11,np.float32);a[9]=.05;a[10]=1;s,_,term,_,_=e.step(a)
   if term or not g(s,'robot','grasp_active'):break
 print('q6',q6off,'term',term,'held',g(s,'robot','grasp_active'),'base',round(g(s,'robot','pos_base_x'),3),'pose',*[round(g(s,'part0','pose_'+q),3) for q in 'xyz'],flush=True);e.close()
