from env_client import make_env
import numpy as np, sys
from kin import Kin, _rz
np.set_printoptions(precision=4, suppress=True, linewidth=200)
env = make_env()
obs, info = env.reset(seed=0)
k = Kin(tool=0.128)
blk = obs[54:57].copy()
def step_to(obs, base_t, q_t, grip, n=40, tol=0.01):
    for i in range(n):
        a = np.zeros(11, np.float32)
        a[0:2] = np.clip(base_t[:2]-obs[16:18], -0.1, 0.1)
        a[2] = np.clip(base_t[2]-obs[18], -0.1, 0.1)
        a[3:10] = np.clip((q_t - obs[19:26] + np.pi) % (2*np.pi) - np.pi, -0.1, 0.1)
        a[10] = grip
        obs, r, te, tr, info = env.step(a)
        err = max(np.abs(obs[16:19]-base_t).max(), np.abs((q_t - obs[19:26] + np.pi) % (2*np.pi) - np.pi).max())
        if err < tol and i > 3: break
    return obs
base = np.array([blk[0]-0.55, blk[1], 0.0])
q = obs[19:26].copy()
obs = step_to(obs, base, q, 0.0)
print("base", obs[16:19])
yaw = 0.0
for dz, g in [(0.15, 0.0), (-0.007, 0.0), (-0.007, 1.0), (0.2, 1.0)]:
    p = blk + np.array([0, 0, dz])
    q, e = k.ik(obs[16:19], obs[19:26], p, _rz(np.pi/2 + yaw))
    obs = step_to(obs, obs[16:19].copy(), q, g, n=60)
    pa, _ = k.fk(obs[16:19], obs[19:26])
    if dz == 0.0 and g == 1.0: obs_desc = obs.copy()
    print(dz, g, "ikerr", e, "fk", pa, "blk", obs[54:57], "q", obs[19:26], "qt", q)
np.save("scratch/obs_calib.npy", obs_desc)
env.close()
