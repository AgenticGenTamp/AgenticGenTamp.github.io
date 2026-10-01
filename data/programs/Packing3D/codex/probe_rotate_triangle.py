from env_client import make_env
from probe_pick_place import JOINTS,OFFSET,g,move
import numpy as np
for angle in (np.pi,-np.pi,np.pi/2,-np.pi/2):
 e=make_env();s,_=e.reset(seed=0,options={'object_count':1});r=s.get_object_from_name('robot');p=s.get_object_from_name('part0');rack=s.get_object_from_name('rack');c=np.array([.03,.03])
 xy=np.array([g(s,p,'pose_x'),g(s,p,'pose_y')]);base=xy-OFFSET+c;s,*_=move(e,s,r,base,grip=1);s,*_=move(e,s,r,base,grip=-1,steps=1)
 lift=JOINTS.copy();lift[1]-=.15;s,*_=move(e,s,r,base,joints=lift,grip=0,steps=1)
 rotated=lift.copy();rotated[6]+=angle;s,*_=move(e,s,r,base,joints=rotated,grip=0,steps=20)
 rackxy=np.array([g(s,rack,'pose_x'),g(s,rack,'pose_y')]);place=rackxy+[0,.07]-OFFSET-[.0235403,0]+c;s,*_=move(e,s,r,place,joints=rotated,grip=0,steps=5)
 term=False
 for _ in range(15):
  a=np.zeros(11,np.float32);a[4]=.02;s,_,term,_,_=e.step(a)
  if term or not g(s,r,'grasp_active'):break
 if not term and g(s,r,'grasp_active'):
  a=np.zeros(11,np.float32);a[6]=-.02;a[10]=1;s,_,term,_,_=e.step(a)
 print('angle',angle,'term',term,'held',g(s,r,'grasp_active'),'q7',g(s,r,'joint_7'),'pose',*[round(g(s,p,'pose_'+q),3) for q in 'xyz'],flush=True);e.close()
