import sys
import numpy as np
from env_client import make_env

pose_id=int(sys.argv[1]) if len(sys.argv)>1 else 0
grip=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
poses=[
 [0,-1.57,3.142,-1.57,0,-1.57,1.57],
 [0,0,3.142,-1.57,0,-1.57,1.57],
 [0,-1.57,3.142,0,0,-1.57,1.57],
 [0,1.0,3.142,-1.0,0,-1.0,1.57],
 [0,0,3.142,0,0,0,1.57],
 [0,-0.5,3.142,0,0,0,1.57],
 [0,0.5,3.142,0,0,0,1.57],
 [0,-1.0,3.142,0,0,0,1.57],
]
target=np.array(poses[pose_id])
e=make_env(); s,_=e.reset(seed=1); o=s.get_object_from_name('cuboid_1'); rob=s.get_object_from_name('robot')
def val(ob,f): return float(s.get(ob,f))
def objp(): return np.array([val(o,f) for f in ('x','y','z')])
orig=objp()
for k in range(120):
 q=np.array([val(rob,f'pos_arm_joint{i}') for i in range(1,8)])
 # angular shortest not used since physical joint representation has useful direct targets
 er=target-q
 a=np.zeros(11,np.float32); a[3:10]=np.clip(er*1.2,-.1,.1); a[10]=1
 s,*_=e.step(a)
 if np.max(abs(er))<.03: break
print('pose',pose_id,'steps',k,'q',np.round(q,2),flush=True)
# y-align to object, open; then scan in +x from initial, with closed gripper.
for k in range(4):
 a=np.zeros(11,np.float32); a[1]=.065; a[10]=1; s,*_=e.step(a)
for k in range(22):
 # Move while opposite/open, then explicitly transition to the candidate close
 # state at every x sample; a proximity-based weld may trigger only on closure.
 a=np.zeros(11,np.float32); a[0]=.1; a[10]=1.0-grip; s,r,t,tr,inf=e.step(a)
 a=np.zeros(11,np.float32); a[10]=grip; s,r,t,tr,inf=e.step(a)
 p=objp(); b=np.array([val(rob,'pos_base_x'),val(rob,'pos_base_y')])
 if k%5==0: print(k,'b',np.round(b,2),'p',np.round(p,3),flush=True)
 if np.linalg.norm(p-orig)>.003: print('HIT',k,'base',b,'p',p,flush=True); break
e.close()
