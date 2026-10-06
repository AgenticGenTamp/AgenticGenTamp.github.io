import sys, numpy as np
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
from approach import obs_to_flat, quat_yaw
k=int(sys.argv[1]); env=E([k],[k])
for s in [int(x) for x in sys.argv[2].split(',')]:
    o,_=env.reset(seed=s, options={'object_count':k}); f,_=obs_to_flat(o)
    print(s, ' '.join(f"({f[39+11*i]:.2f},{f[40+11*i]:.2f},{np.degrees(quat_yaw(f[42+11*i:46+11*i])):.0f})" for i in range(k)))
