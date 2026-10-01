from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
J=['pos_arm_joint%d'%i for i in range(1,8)]
def js(o): return np.array([float(o.get(R,f)) for f in J])
def g(o): return round(float(o.get(R,'pos_gripper')),4), round(float(o.get(R,'vel_gripper')),4)
A = lambda: np.zeros(11, dtype=np.float32)
for j in range(7):
    obs,_=env.reset(seed=0); j0=js(obs)
    a=A(); a[3+j]=0.1
    obs,*_=env.step(a); d1=js(obs)-j0
    tr=[round((js(obs)-j0)[j],4)]
    for i in range(4):
        obs,*_=env.step(A()); tr.append(round((js(obs)-j0)[j],4))
    print('joint',j+1,'+0.1 then zeros: own delta traj',tr,'other max',round(np.max(np.abs(np.delete(d1,j))),4))
# repeated steps joint2 +0.1 x5
obs,_=env.reset(seed=0); j0=js(obs)
a=A(); a[4]=0.1
for i in range(5):
    obs,*_=env.step(a); print('j2 cumulative', round((js(obs)-j0)[1],4))
# joint limit test: joint4 push + for many steps (j4 init -2.548)
obs,_=env.reset(seed=0)
a=A(); a[6]=-0.1
for i in range(15):
    obs,*_=env.step(a)
print('j4 after 15 x -0.1:', round(js(obs)[3],4))
a=A(); a[4]=0.1
for i in range(40):
    obs,*_=env.step(a)
print('j2 after 40 x +0.1:', round(js(obs)[1],4))
# joint1 wrap
obs,_=env.reset(seed=0)
a=A(); a[3]=0.1
for i in range(40): obs,*_=env.step(a)
print('j1 after 40 x +0.1:', round(js(obs)[0],4))
# gripper
obs,_=env.reset(seed=0)
a=A(); a[10]=1.0
for i in range(12):
    obs,*_=env.step(a); print('grip=1 step',i,g(obs))
a=A(); a[10]=0.0
for i in range(8):
    obs,*_=env.step(a); print('grip=0 step',i,g(obs))
a=A(); a[10]=0.5
for i in range(8):
    obs,*_=env.step(a); print('grip=.5 step',i,g(obs))
