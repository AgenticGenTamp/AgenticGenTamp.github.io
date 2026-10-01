from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
J=['pos_arm_joint%d'%i for i in range(1,8)]
def q(o): return np.array([o.get(R,j) for j in J])
def v(o): return np.array([o.get(R,'vel_arm_joint%d'%i) for i in range(1,8)])
q0=q(obs)
for k in range(15):
    a=np.zeros(11,np.float32); a[3]=0.1; a[4]=0.05; a[9]=-0.1
    obs,*_=env.step(a); print('c',k,np.round(q(obs)-q0,3), np.round(v(obs)[[0,1,6]],2))
for k in range(4):
    obs,*_=env.step(np.zeros(11,np.float32)); print('z',np.round(q(obs)-q0,3))
q0=q(obs)
a=np.zeros(11,np.float32); a[3]=0.1
obs,*_=env.step(a); print('imp',np.round(q(obs)-q0,3))
for k in range(4):
    obs,*_=env.step(np.zeros(11,np.float32)); print('z',np.round(q(obs)-q0,3))
