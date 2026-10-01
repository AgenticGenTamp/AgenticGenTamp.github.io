"""Search a type-1 grasp point that seats at the positive rack row."""
from env_client import make_env
from probe_pick_place import JOINTS, OFFSET, g, move
import numpy as np

grid=[0,.015,-.015,.03,-.03,.045,-.045]
for dx,dy in [(x,y) for x in grid for y in grid]:
 env=make_env();s,_=env.reset(seed=0,options={'object_count':1});r=s.get_object_from_name('robot');p=s.get_object_from_name('part0');rack=s.get_object_from_name('rack')
 xy=np.array([g(s,p,'pose_x'),g(s,p,'pose_y')]);c=np.array([dx,dy]);base=xy-OFFSET+c
 s,*_=move(env,s,r,base,grip=1,steps=5);s,*_=move(env,s,r,base,grip=-1,steps=1)
 if not g(s,r,'grasp_active'):env.close();continue
 lift=JOINTS.copy();lift[1]-=.15;s,*_=move(env,s,r,base,joints=lift,grip=0,steps=1)
 rackxy=np.array([g(s,rack,'pose_x'),g(s,rack,'pose_y')]);place=rackxy+[0,.07]-OFFSET-[.0235403,0]+c
 s,*_=move(env,s,r,place,joints=lift,grip=0,steps=5);term=False
 for i in range(15):
  a=np.zeros(11,np.float32);a[4]=.02;s,_,term,_,_=env.step(a)
  if term or not g(s,r,'grasp_active'):break
 if not term and g(s,r,'grasp_active'):
  a=np.zeros(11,np.float32);a[6]=-.02;a[10]=1;s,_,term,_,_=env.step(a)
 print('corr',dx,dy,'term',term,'held',g(s,r,'grasp_active'),'pose',*[round(g(s,p,'pose_'+q),3) for q in 'xyz'],flush=True)
 env.close()
 if term:break
