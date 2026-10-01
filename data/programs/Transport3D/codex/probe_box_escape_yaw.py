import numpy as np
from env_client import make_env
Q=np.array([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593])
def v(s,n,f):return float(s.get(s.get_object_from_name(n),f))
def go(e,s,base=None,q=None,yaw=None,n=40):
 for _ in range(n):
  a=np.zeros(11,np.float32)
  if base is not None:a[:2]=np.clip(np.array(base)-[v(s,'robot','pos_base_x'),v(s,'robot','pos_base_y')],-.2,.2)
  if yaw is not None:a[2]=np.clip(yaw-v(s,'robot','pos_base_rot'),-.2,.2)
  if q is not None:a[3:10]=np.clip(q-np.array([v(s,'robot','joint_'+str(i)) for i in range(1,8)]),-.2,.2)
  a[10]=1;s=e.step(a)[0]
 return s
for seed,count in [(4,0),(4,1),(9,1)]:
 e=make_env();s,_=e.reset(seed=seed,options={'object_count':count}); t=[v(s,'box0','pose_x')-.799987478,v(s,'box0','pose_y')-.347800459]
 s=go(e,s,q=Q);s=go(e,s,q=Q,yaw=2.10479);s=go(e,s,base=t,q=Q,yaw=2.10479)
 print(seed,count,'base',v(s,'robot','pos_base_x'),v(s,'robot','pos_base_y'),'yaw',v(s,'robot','pos_base_rot'))
 e.close()
