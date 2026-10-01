from env_client import make_env
import numpy as np
import math

e=make_env();o,_=e.reset(seed=1);r=o.get_object_from_name('robot');b=o.get_object_from_name('green1');xy=np.array([o.get(b,'pose_x'),o.get(b,'pose_y')])
def move(pose,close=False):
 global o
 for _ in range(60):
  fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)];cur=np.array([o.get(r,f) for f in fs]);d=np.array(pose)-cur
  for i in [2,7,9]:d[i]=(d[i]+math.pi)%(2*math.pi)-math.pi
  if np.max(np.abs(d))<1e-4:return True
  a=np.zeros(11);a[:10]=np.clip(d,-.2,.2);a[-1]=-1 if close else 1;n,*_=e.step(a);nxt=np.array([n.get(r,f) for f in fs]);o=n
  if np.linalg.norm(nxt-cur)<1e-5:return False
 return False
for lift in [.9,1.,.7,.5]:
 q=[0,lift,0,-1.2,math.pi,lift-1.2,math.pi]
 for dy in [.188,.118,.258]:
  o,_=e.reset(seed=1);pose=[xy[0]+1.5,xy[1]+dy,math.pi]+q
  move(pose)
  for dx in np.arange(1.3,.39,-.025):
   pose[:2]=[xy[0]+dx,xy[1]+dy]
   a=np.zeros(11);a[-1]=1;o,*_=e.step(a)
   ok=move(pose,True)
   if o.get(r,'grasp_active'):
    print('GRASP',lift,dx,dy,[o.get(r,f) for f in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]+['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']],flush=True);e.close();raise SystemExit
   if not ok:
    print('COLLISION',lift,dy,dx,[round(float(o.get(r,f)),3) for f in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]],flush=True)
    break
 print('LIFT FAILED',lift,flush=True)
e.close()
