from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name("robot")
def rs(o):
    return [round(float(o.get(R,f)),4) for f in ["pos_base_x","pos_base_y","pos_base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7","finger_state","grasp_active"]]
print("init", rs(obs))
a = np.zeros(11, dtype=np.float32)
# test base x
a[0]=0.2
for i in range(3):
    obs,r,t,tr,info = env.step(a)
    print("bx", rs(obs), r, t, tr)
a[:]=0; a[2]=0.2
for i in range(2):
    obs,r,t,tr,info=env.step(a); print("brot", rs(obs))
a[:]=0; a[0]=0.2
obs,r,t,tr,info=env.step(a); print("bx after rot", rs(obs))
a[:]=0; a[3]=0.2; a[4]=0.2
for i in range(3):
    obs,r,t,tr,info=env.step(a); print("j", rs(obs))
a[:]=0; a[10]=-1
obs,r,t,tr,info=env.step(a); print("close", rs(obs))
a[:]=0; a[10]=1
obs,r,t,tr,info=env.step(a); print("open", rs(obs))
# drive joints far to hit floor
a[:]=0; a[4]=0.2
for i in range(20):
    obs,r,t,tr,info=env.step(a)
print("after many j2", rs(obs))
env.close()
