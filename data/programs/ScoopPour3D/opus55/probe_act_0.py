from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
print("info", info)
R = obs.get_object_from_name('robot')
F = ['pos_base_x','pos_base_y','pos_base_rot']+[f'pos_arm_joint{i}' for i in range(1,8)]+['pos_gripper']
def st(o): return np.array([float(o.get(R,f)) for f in F])
print("init", np.round(st(obs),4))
for t in range(30):
    obs,r,te,tr,info = env.step(np.zeros(11))
    print(t, r, te, tr, info if t<2 else '', np.round(st(obs),4))
