import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from env_client import make_env

def fk(q):
 p=np.array([0.,.188,0.]); r=np.eye(3)
 for a,axis,off in zip(q,['z','y','x','y','x','y','x'],[[0,0,0],[.1,0,0],[0,0,0],[.4,0,0],[0,0,0],[.321,0,0],[0,0,0]]):
  p+=r@off; r=r@R.from_euler(axis,a).as_matrix()
 return p+r@np.array([.18,0,0]),r
q0=np.array([.67717022,-.34313199,1.2,-1.46688402,1.24223232,-1.95442832,2.22254133])
rd=fk(q0)[1]
lo=np.array([-2.28,-.52359,-3.9,-2.32,-10,-2.18,-10]);hi=np.array([.714,1.396,3.9,0,10,0,10])
e=make_env()
for z in np.arange(.12,-.401,-.02):
 s,_=e.reset(seed=0);rob=s.get_object_from_name('robot');b=s.get_object_from_name('block1');bpos=np.array([s.get(b,'pose_'+k) for k in 'xyz']);pos=np.array([.72,.2,z])
 def err(q):
  p,r=fk(q);return np.r_[p-pos,.25*R.from_matrix(rd.T@r).as_rotvec()]
 sol=least_squares(err,q0,bounds=(lo,hi),max_nfev=120,ftol=1e-8)
 if np.linalg.norm(err(sol.x))>.01:print('IK fail',z,flush=True);continue
 target=np.r_[bpos[0]-.72,bpos[1]-.2,0,sol.x]
 def getq():return np.array([s.get(rob,k) for k in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]])
 for phase in range(2):
  t=target.copy()
  if phase==0:t[0]=-1
  for _ in range(20):
   prev=getq();act=np.zeros(11);act[:10]=np.clip(t-prev,-.15,.15);act[10]=1
   s,*rest=e.step(act.astype(np.float32))
   if s.get(rob,'grasp_active'):
    print('SUCCESS z',z,'q',getq().tolist(),'tf',[s.get(rob,'grasp_tf_'+k) for k in ['x','y','z','qx','qy','qz','qw']],flush=True);e.close();raise SystemExit
   if np.linalg.norm(getq()-prev)<1e-6 or np.linalg.norm(t-getq())<1e-4:break
 s,*rest=e.step(np.array([0]*10+[-1],dtype=np.float32))
 if s.get(rob,'grasp_active'):
  print('SUCCESS CLOSED z',z,'q',getq().tolist(),'tf',[s.get(rob,'grasp_tf_'+k) for k in ['x','y','z','qx','qy','qz','qw']],flush=True);e.close();raise SystemExit
 print('z',z,'err',np.linalg.norm(err(sol.x)),'got',getq().tolist(),flush=True)
e.close()
