from env_client import make_env
import numpy as np
import math

e=make_env();o,_=e.reset(seed=0)
r=o.get_object_from_name('robot'); b=o.get_object_from_name('blocker'); target=np.array([o.get(b,'pose_x'),o.get(b,'pose_y')]); qz=o.get(b,'pose_qz');qw=o.get(b,'pose_qw'); theta=2*math.atan2(qz,qw)
def move(target,close=False):
 global o
 for _ in range(50):
  cur=np.array([o.get(r,f) for f in ['base_x','base_y','base_rot']]);d=np.array(target)-cur;d[2]=(d[2]+math.pi)%(2*math.pi)-math.pi
  if np.max(np.abs(d))<1e-5: return True
  a=np.zeros(11);a[:3]=np.clip(d,-.2,.2);a[-1]=-1 if close else 1
  n,*_=e.step(a)
  nxt=np.array([n.get(r,f) for f in ['base_x','base_y','base_rot']]);o=n
  if np.linalg.norm(nxt-cur)<1e-8: return False
 return False
# start to side away from table then sweep x approach on y rows
for angle in [theta,0,theta-math.pi/2,theta+math.pi/2,math.pi]:
 for by in np.arange(-1.2,1.25,.08):
  o,_=e.reset(seed=0)
  move([2.5,by,angle])
  for bx in np.arange(3.0,4.15,.025):
   moved=move([bx,by,angle],True)
   if o.get(r,'grasp_active'):
    print('GRASP',angle,bx,by,'robot', [o.get(r,f) for f in ['base_x','base_y','base_rot']], 'block',[(n,o.get(o.get_object_from_name(n),'grasp_active')) for n in ['blocker','green0']],flush=True);e.close();raise SystemExit
   if not moved: break
 print('ANGLE FAILED',angle,flush=True)
e.close()
