import numpy as np
from env_client import make_env

env=make_env(); s,_=env.reset(seed=0)
r=s.get_object_from_name('robot')
f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
def vals():return np.array([s.get(r,k) for k in f])
def step(a):
 global s
 s,*_=env.step(np.array(a,dtype=np.float32))
 return s.get(r,'grasp_active')
def move(t,close=True):
 for _ in range(30):
  v=vals(); a=np.zeros(11); a[:10]=np.clip(t-v,-.2,.2); a[10]=-1 if close else 1
  active=step(a)
  if active:return True
  if np.max(np.abs(vals()-v))<1e-6 or np.max(np.abs(t-vals()))<1e-4:break
 return False
q0=vals(); print('start',q0,flush=True)
print('blocks',[(o.name,[s.get(o,'pose_'+k) for k in 'xyz']) for o in s.get_objects(env.observation_space.get_type('block'))],flush=True)
for pitch in [0,.2,-.2,.4,-.4,.6,-.6]:
 s,_=env.reset(seed=0)
 t=q0.copy(); t[4]+=pitch
 move(t,False)
 for iy,y in enumerate(np.arange(-.8,.85,.08)):
  for x in np.arange(-1.1,-.29,.04)[::1 if iy%2==0 else -1]:
   t[:2]=[x,y]
   if move(t):
    print('SUCCESS',vals().tolist(),'tf',[s.get(r,'grasp_tf_'+k) for k in ['x','y','z','qx','qy','qz','qw']], 'held',[(o.name,[s.get(o,'pose_'+k) for k in 'xyz']) for o in s.get_objects(env.observation_space.get_type('block')) if s.get(o,'grasp_active')],flush=True)
    env.close();raise SystemExit
 print('pitch failed',pitch,'at',vals(),flush=True)
env.close()
