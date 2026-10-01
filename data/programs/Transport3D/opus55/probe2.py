from env_client import make_env
import numpy as np
env = make_env()
R=['pos_base_x','pos_base_y','pos_base_rot','joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7','finger_state','grasp_active']
def rs(obs):
    r=obs.get_object_from_name('robot'); return [round(float(obs.get(r,f)),3) for f in R]
obs,_=env.reset(seed=1)
print(rs(obs))
def st(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.array(a,dtype=np.float32)); print(rs(obs),rew,term,trunc)
st([0.2,0,0]+[0]*8)
st([0,0,0.2]+[0]*8)
st([0.2,0,0]+[0]*8)
st([0.5,0,0]+[0]*8)
st([0]*3+[0.1,0,0,0,0,0,0,0])
st([0]*3+[0,0.1,0,0,0,0,0,0])
st([0]*10+[-1])
st([0]*10+[0])
st([0]*10+[1])
for i in range(20): st([0.2,0,0]+[0]*8)
env.close()
