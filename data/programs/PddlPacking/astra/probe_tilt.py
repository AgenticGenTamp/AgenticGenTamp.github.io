import numpy as np
from scipy.spatial.transform import Rotation as R
from env_client import make_env
from kinematics import ik,fk,Q0
import sys
for seed in [int(x) for x in sys.argv[1:]] or [0]:
 for pitch in [.7]:
  e=make_env();s,_=e.reset(seed=seed);rob=s.get_object_from_name('robot');blocks=[s.get_object_from_name(n) for n in s.get_object_names() if n.startswith('block')];b=max(blocks,key=lambda b:s.get(b,'pose_x'));pos=np.array([s.get(b,'pose_'+k) for k in 'xyz']);base=np.array([-.66,pos[1]-.25,0]);rot=R.from_euler('y',pitch).as_matrix();tool=pos-.04*rot[:,0];steps=0
  def conf():return np.array([s.get(rob,k) for k in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]])
  def move(t,grip=1):
   global s,steps
   for _ in range(80):
    c=conf();d=t-c
    for j in [2,7,9]:d[j]=(d[j]+np.pi)%(2*np.pi)-np.pi
    if max(abs(d))<.002: return True
    a=np.r_[d*min(1,.15/max(abs(d))),grip];s,*_=e.step(a.astype(np.float32));steps+=1
    if max(abs(conf()-c))<1e-6:return False
   return False
  ok=move(np.r_[base,Q0]);q=Q0
  for z in [1.1,1.04,.98,.92,.87,tool[2]]:
   p=tool-np.r_[base[:2],0];p[2]=z;q=ik(p,q0=q,rot=rot);ok=move(np.r_[base,q]);print(seed,pitch,'z',round(z,3),'ok',ok,'err',round(np.linalg.norm(fk(conf()[3:])[0]-p),3),flush=True)
   if not ok:break
  a=np.zeros(11);a[-1]=-1;s,*_=e.step(a.astype(np.float32));print('RESULT',seed,pitch,pos.tolist(),'held',s.get(rob,'grasp_active'),'steps',steps,'toolworld',(fk(conf()[3:])[0]+np.r_[conf()[:2],0]).tolist(),flush=True);e.close()
