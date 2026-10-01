from env_client import make_env
import numpy as np, json
env = make_env()
obs, info = env.reset(seed=0)
def rj(obs):
    r = obs.get_object_from_name('robot')
    return np.array([float(obs.get(r,f)) for f in ['pos_base_x','pos_base_y','pos_base_rot','joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7','finger_state','grasp_active']])
print("init", rj(obs))
# base only
for idx in [0,1,2]:
    obs,info = env.reset(seed=0)
    a=np.zeros(11,dtype=np.float32); a[idx]=0.2
    for i in range(3):
        obs,rew,t,tr,info=env.step(a)
    print("idx",idx, rj(obs)[:3])
# joints one at a time, positive then negative, find range
for j in range(3,10):
    for sgn in [1,-1]:
        obs,info=env.reset(seed=0)
        a=np.zeros(11,dtype=np.float32); a[j]=0.2*sgn
        prev=rj(obs)[j]; n=0
        for i in range(40):
            obs,rew,t,tr,info=env.step(a)
            cur=rj(obs)[j]
            if abs(cur-prev)<1e-6: break
            prev=cur; n+=1
        print("joint",j-2,"sgn",sgn,"steps",n,"final",round(prev,3))
# gripper
obs,info=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32); a[10]=-1
obs,rew,t,tr,info=env.step(a); print("close",rj(obs)[10:])
a=np.zeros(11,dtype=np.float32); a[10]=1
obs,rew,t,tr,info=env.step(a); print("open",rj(obs)[10:])
