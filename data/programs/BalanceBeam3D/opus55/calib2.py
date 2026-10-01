from env_client import make_env
import numpy as np, sys
from kin import Kin, _rz
np.set_printoptions(precision=4, suppress=True, linewidth=200)
env = make_env()
obs, info = env.reset(seed=0)
k = Kin()
def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi
def step_to(obs, base_t, q_t, grip, n=40, tol=0.01):
    for i in range(n):
        a = np.zeros(11, np.float32)
        a[0:2] = np.clip(base_t[:2]-obs[16:18], -0.1, 0.1)
        a[2] = np.clip(base_t[2]-obs[18], -0.1, 0.1)
        a[3:10] = np.clip(wrap(q_t - obs[19:26]), -0.1, 0.1)
        a[10] = grip
        obs, r, te, tr, info = env.step(a)
        err = max(np.abs(obs[16:19]-base_t).max(), np.abs(wrap(q_t - obs[19:26])).max())
        if err < tol and i > 3: break
    return obs
base = np.array([0.15, 0.206, 0.0])
obs = step_to(obs, base, obs[19:26].copy(), 1.0)
tgt = np.array([0.70, 0.0, 0.15])
for z in np.arange(0.15, -0.12, -0.01):
    tgt[2] = z
    q, e = k.ik(obs[16:19], obs[19:26], tgt, _rz(np.pi/2))
    obs = step_to(obs, obs[16:19].copy(), q, 1.0, n=25, tol=0.003)
    pa, _ = k.fk(obs[16:19], obs[19:26])
    print("%.3f"%z, "fk", pa, "qerr", wrap(q-obs[19:26]))
env.close()
