from env_client import make_env
import numpy as np, time
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
names=sorted(obs.get_object_names()); print(names)
J=['pos_arm_joint%d'%i for i in range(1,8)]
def st(o): return np.array([o.get(R,'pos_base_x'),o.get(R,'pos_base_y'),o.get(R,'pos_base_rot')]+[o.get(R,j) for j in J]+[o.get(R,'pos_gripper')])
for n in names:
    if n.startswith('cube') or n.startswith('wiper'):
        o=obs.get_object_from_name(n); print(n,[round(obs.get(o,f),3) for f in ['x','y','z']])
print(np.round(st(obs),3))
t=time.time()
for k in range(6):
    a=np.zeros(11,np.float32); a[3]=0.1; a[5]=-0.05
    obs,*_=env.step(a); print('j',np.round(st(obs),3))
for k in range(6):
    a=np.zeros(11,np.float32)
    obs,*_=env.step(a); print('z',np.round(st(obs),3))
print('dt/step',(time.time()-t)/12)
for k in range(5):
    a=np.zeros(11,np.float32); a[0]=0.1
    obs,*_=env.step(a); print('bx',np.round(st(obs),3))
for k in range(5):
    a=np.zeros(11,np.float32); a[2]=0.1
    obs,*_=env.step(a); print('br',np.round(st(obs),3))
for k in range(5):
    a=np.zeros(11,np.float32); a[0]=0.1
    obs,*_=env.step(a); print('bx',np.round(st(obs),3))
for k in range(8):
    a=np.zeros(11,np.float32); a[10]=1
    obs,*_=env.step(a); print('g',np.round(st(obs),3))
