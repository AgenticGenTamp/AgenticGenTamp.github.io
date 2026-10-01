from env_client import make_env
import numpy as np, json
env = make_env()
obs,info = env.reset(seed=0)
def R(o):
    r=o.get_object_from_name("robot")
    return np.array([float(o.get(r,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7","finger_state","grasp_active"]])
print("start",R(obs))
# try small base move
for d in [0.05,0.1,0.4]:
    a=np.zeros(11); a[0]=d
    o2,r,t,tr,i=env.step(a); print("dx",d,R(o2)[:3],"rew",r)
for d in [0.05,0.4]:
    a=np.zeros(11); a[1]=d
    o2,r,t,tr,i=env.step(a); print("dy",d,R(o2)[:3],"rew",r)
for d in [0.05,0.4]:
    a=np.zeros(11); a[2]=d
    o2,r,t,tr,i=env.step(a); print("drot",d,R(o2)[:3],"rew",r)
env.close()
