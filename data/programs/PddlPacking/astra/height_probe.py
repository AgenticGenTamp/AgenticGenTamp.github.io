from env_client import make_env
from kinematics import *
e=make_env();s,_=e.reset(seed=0);rob=s.get_object_from_name('robot');b=s.get_object_from_name('block1')
def conf(s):return np.array([s.get(rob,k) for k in ['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]])
def move(goal,close=0):
 global s
 for i in range(35):
  cur=conf(s);d=np.clip(goal-cur,-.2,.2);s,r,t,tr,info=e.step(np.r_[d,close])
  if np.max(abs(conf(s)-goal))<1e-4: return True
  if np.max(abs(cur-conf(s)))<1e-6: return False
 return False
bx=s.get(b,'pose_x');by=s.get(b,'pose_y');base=np.array([bx-.72+.05,by-.25,0])
move(np.r_[base,Q0],1)
for h in np.arange(.85,1.16,.005):
 yaw=2*np.arctan2(s.get(b,'pose_qz'),s.get(b,'pose_qw'))
 rot=Rotation.from_euler('z',yaw).as_matrix()@DOWN
 q=ik(np.array([.72,.25,.80]),h,rot=rot)
 s,_,_,_,_=e.step(np.r_[np.zeros(10),1])
 ok=move(np.r_[base,q],1)
 s,_,_,_,_=e.step(np.r_[np.zeros(10),-1])
 active=s.get(rob,'grasp_active');print('base',conf(s)[:3].round(3),'h',round(h,3),'ok',ok,'active',active,'open',s.get(rob,'gripper_opening'),'q',conf(s)[3:].round(4),flush=True)
 if active:
  for name in ['robot','block1']:
   o=s.get_object_from_name(name);print(name,{f:s.get(o,f) for f in e.observation_space.type_features[e.observation_space.get_type('robot' if name=='robot' else 'block')]},flush=True)
  break
e.close()
