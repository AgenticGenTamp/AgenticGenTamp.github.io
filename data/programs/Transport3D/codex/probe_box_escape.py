import numpy as np
from env_client import make_env
from approach import GeneratedApproach

Q=np.array([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593])
def v(s,n,f):return float(s.get(s.get_object_from_name(n),f))
def act(e,s,base=None,q=None,n=30):
 for k in range(n):
  a=np.zeros(11,np.float32)
  if base is not None:
   a[:2]=np.clip(np.array(base)-[v(s,'robot','pos_base_x'),v(s,'robot','pos_base_y')],-.2,.2)
  if q is not None:
   cur=np.array([v(s,'robot','joint_'+str(i)) for i in range(1,8)])
   a[3:10]=np.clip(q-cur,-.2,.2)
  a[10]=1;s=e.step(a)[0]
 return s
for seed,count in [(4,0),(4,1),(9,1)]:
 for mode in ['q_base','base_q','x_y_q','back_q_base']:
  e=make_env();s,_=e.reset(seed=seed,options={'object_count':count})
  target=[v(s,'box0','pose_x')-.799987478,v(s,'box0','pose_y')-.347800459]
  if mode=='q_base':s=act(e,s,q=Q);s=act(e,s,base=target,q=Q)
  if mode=='base_q':s=act(e,s,base=target);s=act(e,s,base=target,q=Q)
  if mode=='x_y_q':s=act(e,s,base=[target[0],0]);s=act(e,s,base=target);s=act(e,s,base=target,q=Q)
  if mode=='back_q_base':s=act(e,s,base=[-.5,.5]);s=act(e,s,base=[-.5,.5],q=Q);s=act(e,s,base=target,q=Q)
  print(seed,count,mode,'base',round(v(s,'robot','pos_base_x'),2),round(v(s,'robot','pos_base_y'),2),'q',[round(v(s,'robot','joint_'+str(i)),2) for i in range(1,8)])
  e.close()
