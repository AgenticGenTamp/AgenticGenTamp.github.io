from env_client import make_env
import numpy as np, time
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
def st(o):
    return [round(float(o.get(R,f)),4) for f in ['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint2','pos_arm_joint4','pos_arm_joint6','pos_gripper','vel_base_x']]
print('init', st(obs))
A = lambda: np.zeros(11, dtype=np.float32)
ts=[]
# zero action 5 steps
for i in range(5):
    t=time.time(); obs,r,te,tr,info = env.step(A()); ts.append(time.time()-t)
    print('zero',i,st(obs),r,te,tr,info)
# base x +0.1 for 1 step then zeros
a=A(); a[0]=0.1
obs,r,te,tr,info=env.step(a); print('bx+.1',st(obs),r)
for i in range(6):
    obs,r,te,tr,info=env.step(A()); print(' zero',i,st(obs))
# base x 0.05 several steps
a=A(); a[0]=-0.05
for i in range(4):
    obs,r,te,tr,info=env.step(a); print('bx-.05',i,st(obs))
for i in range(4):
    obs,r,te,tr,info=env.step(A()); print(' zero',i,st(obs))
# base y
a=A(); a[1]=0.1
obs,r,te,tr,info=env.step(a); print('by+.1',st(obs))
for i in range(4):
    obs,r,te,tr,info=env.step(A()); print(' zero',i,st(obs))
# yaw
a=A(); a[2]=0.1
obs,r,te,tr,info=env.step(a); print('rot+.1',st(obs))
for i in range(4):
    obs,r,te,tr,info=env.step(A()); print(' zero',i,st(obs))
print('mean step time', np.mean(ts))
