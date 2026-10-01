from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name("robot")
F = ["base_x","base_y","base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7","gripper_opening","grasp_active"]
def rv(o):
    return np.array([o.get(o.get_object_from_name("robot"), f) for f in F])
def rob(s):
    r = s.get_object_from_name("robot")
    return np.array([s.get(r,f) for f in F])
print("init", np.round(rob(obs),4))
# test single joint deltas
for idx in range(10):
    a = np.zeros(11); a[idx]=0.2
    o2,_,_,_,_ = env.step(a)
    print(idx, np.round(rob(o2)-rob(obs),4))
    obs=o2
# now drive each joint to limit
for idx in range(3,10):
    for k in range(40):
        a=np.zeros(11); a[idx]=0.2
        obs,_,_,_,_=env.step(a)
    hi = rob(obs)[idx]
    for k in range(80):
        a=np.zeros(11); a[idx]=-0.2
        obs,_,_,_,_=env.step(a)
    lo = rob(obs)[idx]
    print("joint",idx,"hi",round(hi,4),"lo",round(lo,4))
env.close()
