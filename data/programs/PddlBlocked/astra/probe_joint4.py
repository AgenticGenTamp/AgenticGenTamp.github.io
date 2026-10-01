from env_client import make_env
import numpy as np
import math

e=make_env();o,_=e.reset(seed=0);r=o.get_object_from_name('robot'); b=o.get_object_from_name('green1'); xy=np.array([o.get(b,'pose_x'),o.get(b,'pose_y')])
def move(pose,close=False):
 global o
 for _ in range(60):
  cur=np.array([o.get(r,f) for f in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]]);d=np.array(pose)-cur
  for i in [2,7,9]:d[i]=(d[i]+math.pi)%(2*math.pi)-math.pi
  if np.max(np.abs(d))<1e-5:return True
  a=np.zeros(11);a[:10]=np.clip(d,-.2,.2);a[-1]=-1 if close else 1
  n,*_=e.step(a);nxt=np.array([n.get(r,f) for f in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]]);o=n
  if np.linalg.norm(nxt-cur)<1e-8:return False
 return False
init=np.array([o.get(r,'joint_'+str(i)) for i in range(1,8)])
for delta in [-.2,.2,-.4,.4,-.6,.6,-.8,.8,1.,1.2,1.4]:
 q=init.copy();q[3]+=delta
 for dy in np.arange(-.7,.71,.07):
  o,_=e.reset(seed=0)
  pose=[xy[0]+1.4,xy[1]+dy,math.pi]+list(q)
  move(pose)
  for dx in np.arange(1.2,.39,-.035):
   pose[:2]=[xy[0]+dx,xy[1]+dy]
   ok=move(pose,True)
   if o.get(r,'grasp_active'):
    print('GRASP',delta,dx,dy,[o.get(r,f) for f in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]+['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']],flush=True);e.close();raise SystemExit
   if not ok:break
 print('DELTA FAILED',delta,flush=True)
e.close()
