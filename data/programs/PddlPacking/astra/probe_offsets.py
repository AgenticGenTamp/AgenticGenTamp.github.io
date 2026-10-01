from env_client import make_env
from kinematics import *
e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot');b=s.get_object_from_name('block1');bx=s.get(b,'pose_x');by=s.get(b,'pose_y')
f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
def v():return np.array([s.get(r,k) for k in f])
yaw=2*np.arctan2(s.get(b,'pose_qz'),s.get(b,'pose_qw'));rot=Rotation.from_euler('z',yaw).as_matrix()@DOWN
qs=[ik(np.array([.72,.25,z]),rot=rot) for z in np.arange(1.0,.59,-.01)]
for dx in [.05,0,.1,-.05,-.1]:
 for dy in [0,.05,-.05,.1,-.1]:
  s,_=e.reset(seed=0);base=np.array([bx-.72+dx,by-.25+dy,0])
  for iz,q in enumerate(qs):
   s,*_=e.step(np.r_[np.zeros(10),1]);t=np.r_[base,q]
   for _ in range(20):
    prev=v();s,*_=e.step(np.r_[np.clip(t-prev,-.15,.15),1])
    if np.linalg.norm(t-v())<1e-5 or np.linalg.norm(v()-prev)<1e-5:break
   s,*_=e.step(np.r_[np.zeros(10),-1])
   if s.get(r,'grasp_active'):
    print('SUCCESS',dx,dy,'z',1-.01*iz,'q',v().tolist(),'tf',[s.get(r,'grasp_tf_'+k) for k in ['x','y','z','qx','qy','qz','qw']],flush=True);e.close();raise SystemExit
  print('offset',dx,dy,'lowest fk',fk(v()[3:])[0],flush=True)
e.close()
