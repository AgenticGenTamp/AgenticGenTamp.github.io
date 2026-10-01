from env_client import make_env
from kin import *
import numpy as np, sys
np.set_printoptions(precision=4,suppress=True)
env = make_env()
pts = [tuple(map(float,p.split(','))) for p in sys.argv[2:]]
seed=int(sys.argv[1]); YAW=float(__import__("os").environ.get("YAW","0"))
def q_of(o): return np.array([o.get(R,f'joint_{i}') for i in range(1,8)])
def base_of(o): return (o.get(R,'pos_base_x'),o.get(R,'pos_base_y'),o.get(R,'pos_base_rot'))
for (x,y) in pts:
    obs, info = env.reset(seed=seed)
    R = obs.get_object_from_name('robot')
    last=None
    for z in np.arange(0.30, 0.0, -0.005):
        qt,e = ik(base_of(obs), q_of(obs), np.array([x,y,z]), down_R(YAW))
        ok=True
        for _ in range(30):
            q=q_of(obs); d=qt-q
            if np.max(np.abs(d))<1e-4: break
            a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-0.2,0.2)
            obs2,*_=env.step(a)
            if np.allclose(q_of(obs2),q): ok=False; break
            obs=obs2
        if not ok: break
        last=z
    print(f'({x},{y}) lowest reached ee z={last:.3f}, e={e:.1e}')
