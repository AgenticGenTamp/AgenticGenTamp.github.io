from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
def rs(o): return np.array([o.get(R,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'pos_arm_joint{i}' for i in range(1,8)]+['pos_gripper']])
print(rs(obs).round(3))
for k in range(12):
    a = np.zeros(11); 
    if k<3: a[0]=0.1
    elif k<6: a[3]=0.1
    elif k<9: a[10]=1.0
    obs,r,te,tr,inf = env.step(a)
    print(k, round(r,4), te, tr, rs(obs).round(3), inf)
