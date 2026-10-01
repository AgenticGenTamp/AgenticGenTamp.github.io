import numpy as np, math, time
from expD_lib import *
from expD_core import *
env = make_env()
t0=time.time()
for seed in [1,2]:
    qh = home_q(env, seed)
    for mode in ["direct","high"]:
        ap, obs, n = remove_blocker(env, seed)
        b = ap._blocks(obs); g0 = b["green0"]; d = np.array(ap.dir)
        bt = base_for(g0, d, 0.85, -0.188)
        obs, o = approach_grasp(env, ap, obs, g0, d, bt, qh, mode=mode)
        g0o = obs.get_object_from_name("green0")
        ga = float(obs.get(g0o,"grasp_active"))
        print(f"seed={seed} {mode:6s} base={np.round(bt,3)} {o['base_why']:12s} pre_ok={o['pre_ok']} "
              f"pre_err={o['pre_err']} gap={o['gap']} grasped={o['grasped']} g0act={ga} steps={o['steps']}")
print("t=%.1fs"%(time.time()-t0))
