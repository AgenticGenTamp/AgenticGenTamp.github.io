from env_client import make_env
import numpy as np, sys
from kin import Kin, _rz
np.set_printoptions(precision=4, suppress=True, linewidth=200)
def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi
K, KI = float(sys.argv[1]), float(sys.argv[2])
env = make_env()
obs, info = env.reset(seed=0)
k = Kin(mount=(0.113,0,0.36), tool=0.128)
base = obs[16:19].copy()
for tgt in [np.array([0.65, 0.0, 0.10]), np.array([0.65, 0.2, 0.03]), np.array([0.5,-0.1,0.25])]:
    q, e = k.ik(base, obs[19:26], tgt, _rz(np.pi/2))
    I = np.zeros(7)
    for t in range(40):
        eq = wrap(q - obs[19:26])
        I = np.clip(I + KI*eq, -0.1, 0.1)
        a = np.zeros(11, np.float32); a[:3] = np.clip(base - obs[16:19], -.1, .1)
        a[3:10] = np.clip(K*eq + I, -0.1, 0.1)
        obs, *_ = env.step(a)
        p, _ = k.fk(obs[16:19], obs[19:26])
        if t % 3 == 0 or np.linalg.norm(p-tgt) < 0.005:
            print(t, "cart err %.4f" % np.linalg.norm(p - tgt), "qerr %.3f" % np.abs(eq).max(), "vel %.3f" % np.abs(obs[30:37]).max())
        if np.linalg.norm(p-tgt) < 0.005 and np.abs(obs[30:37]).max() < 0.1: break
    print("----", t)
