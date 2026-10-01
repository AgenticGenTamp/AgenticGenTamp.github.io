from env_client import make_env
import numpy as np
E=make_env(); s,_=E.reset(seed=14); r=s.get_objects(E.observation_space.get_type('mujoco_tidybot_robot'))[0]
fs=['pos_arm_joint'+str(i) for i in range(1,8)]
def q(s): return np.array([s.get(r,f) for f in fs])
q0=q(s); print('q0',q0.tolist(),flush=True)
for sign in [1,0,-1,0]:
    old=q(s)
    for k in range(10):
        a=np.zeros(11); a[3:10]=sign*.1
        s,re,te,tr,inf=E.step(a)
        if k in [0,1,2,4,9]: print(sign,k,'delta',np.round(q(s)-old,6).tolist(),'total',np.round(q(s)-q0,6).tolist(),flush=True)
E.close()
