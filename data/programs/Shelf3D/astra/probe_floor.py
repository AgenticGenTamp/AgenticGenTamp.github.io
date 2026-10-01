from env_client import make_env
from kinematics import *
E=make_env();s,info=E.reset(seed=42);r=s.get_object_from_name('robot')
for z in [.2,.1,0,-.1,-.2,-.3,-.4]:
 def res(x):
  q=HOME.copy();q[[1,3,5]]=x;p,R=fk(q)
  return np.r_[(p-np.array([.4,.001,z]))[[0,2]],R[0,2]]
 q=HOME.copy();q[[1,3,5]]=least_squares(res,[1.5,-1.5,0],max_nfev=100).x
 for i in range(120):
  cur=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)])
  a=np.r_[0,0,0,np.clip((q-cur)*2,-.1,.1),0].astype(np.float32)
  s,*_=E.step(a)
 actual=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)])
 print('z',z,'target',q.round(3),'actual',actual.round(3),'pos',fk(actual)[0].round(3),flush=True)
E.close()
