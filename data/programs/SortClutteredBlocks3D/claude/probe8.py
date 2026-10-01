from env_client import make_env
import numpy as np
F=['pos_base_x','pos_base_y','pos_base_rot']+[f'pos_arm_joint{i}' for i in range(1,8)]+['pos_gripper']
for d in range(11):
    env = make_env(); obs,info = env.reset(seed=0)
    o=obs.get_object_from_name('robot')
    start=np.array([float(obs.get(o,f)) for f in F])
    a=np.zeros(11,dtype=np.float32); a[d]= 0.1 if d<10 else 1.0
    for t in range(30):
        obs,rew,term,trunc,info=env.step(a)
    o=obs.get_object_from_name('robot')
    end=np.array([float(obs.get(o,f)) for f in F])
    diff=end-start
    print(d, " ".join(f"{F[i][4:]}:{diff[i]:+.3f}" for i in range(11) if abs(diff[i])>0.005))
    env.close()
