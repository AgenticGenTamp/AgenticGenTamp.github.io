from env_client import make_env
from kinematics import ik,fk
import numpy as np
import sys,time

def vals(s,obj,features):return np.array([s.get(obj,f) for f in features])
e=make_env();s,_=e.reset(seed=0)
rob=s.get_object_from_name('robot'); bn=s.get_object_from_name('bin_yellow_0'); scoop=s.get_object_from_name('scoop_0')
features=['pos_arm_joint'+str(i) for i in range(1,8)]
base=vals(s,rob,['pos_base_x','pos_base_y','pos_base_rot'])
bp=vals(s,bn,['x','y','z']);grasp=float(sys.argv[1]) if len(sys.argv)>1 else 1.
bp=bp+[-.22,0,0]
phases=[(bp+[0,0,.19],1.-grasp,130),(bp+[0,0,.03],1.-grasp,160),(bp+[0,0,.03],grasp,30),(bp+[0,0,.24],grasp,140),(bp+[0,.40,.24],grasp,160),(bp+[0,.40,.07],grasp,100),(bp+[0,.40,.07],1.-grasp,20)]
start=time.time()
for phase,(target,grip,n) in enumerate(phases):
 qtarget,err=ik(target,np.diag([1.,-1.,-1.]),vals(s,rob,features),base)
 print('PHASE',phase,'target',target,'q',qtarget.round(3),'err',err,flush=True)
 for j in range(n):
  if phase in [1,2] and j%15==0:
   target=vals(s,bn,['x','y','z'])+[-.22,0,.03]
   qtarget,err=ik(target,np.diag([1.,-1.,-1.]),vals(s,rob,features),base)
  a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip(2.5*(qtarget-vals(s,rob,features))+.15*vals(s,rob,['vel_arm_joint'+str(k) for k in range(1,8)]),-.1,.1);a[10]=grip
  s,r,t,tr,info=e.step(a)
  if j%50==0 or j==n-1:
   cubes=[o for o in s.get_objects(e.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube_')]
   cp=np.array([vals(s,o,['x','y','z']) for o in cubes]);qp=vals(s,rob,features)
   print(phase,j,'r',r,'ee',fk(qp,base)[:3,3].round(3),'bin',vals(s,bn,['x','y','z']).round(3),'cubes',np.mean(cp,axis=0).round(3),'qerr',round(np.linalg.norm(qtarget-qp),3),flush=True)
  if t or tr:break
print('SECONDS',time.time()-start,flush=True)
e.close()
