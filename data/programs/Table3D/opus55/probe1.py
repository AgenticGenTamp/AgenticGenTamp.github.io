from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_objects(env.observation_space.get_type('Kinematic3DRobot'))[0] if hasattr(env.observation_space,'get_type') else None
names=['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]+['finger_state','grasp_active']
def rs(o):
    r=o.get_object_from_name('robot'); return np.array([o.get(r,n) for n in names])
prev=rs(obs)
tests=[]
for i in range(10):
    a=np.zeros(11,dtype=np.float32); a[i]=0.1; tests.append(a)
a=np.zeros(11,dtype=np.float32); a[3]=0.5; tests.append(a)
a=np.zeros(11,dtype=np.float32); a[10]=-1; tests.append(a)
a=np.zeros(11,dtype=np.float32); a[10]=1; tests.append(a)
a=np.zeros(11,dtype=np.float32); a[2]=0.3; tests.append(a)
a=np.zeros(11,dtype=np.float32); a[0]=0.1; tests.append(a)
for a in tests:
    obs,r,term,trunc,info=env.step(a)
    cur=rs(obs); print(np.nonzero(a)[0], np.round(cur-prev,4), r, term, trunc)
    prev=cur
print(np.round(prev,3))
env.close()
