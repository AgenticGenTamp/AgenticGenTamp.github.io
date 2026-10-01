from env_client import make_env
import numpy as np
np.set_printoptions(precision=4, suppress=True)
names="robot_x robot_y robot_th base_r arm_joint arm_len vac grip_h grip_w hook_x hook_y hook_th hook_static hr hg hb hz hook_w hook_l1 hook_l2 mb_x mb_y mb_th mb_static mr mg mb mz mb_rad tb_x tb_y tb_th tb_static tr tg tb tz tb_rad".split()
env=make_env()
for s in [42,0,1,2]:
    obs,info=env.reset(seed=s)
    print("seed",s, info)
    for n,v in zip(names,obs): print(f"  {n}={v:.4f}")
env.close()
