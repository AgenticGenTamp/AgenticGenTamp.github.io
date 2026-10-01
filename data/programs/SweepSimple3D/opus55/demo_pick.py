from env_client import make_env
import numpy as np, rutil
from rutil import Bot
env = make_env(); obs, _ = env.reset(seed=0); b = Bot(env, obs)
for name, dst in [('cube_1', (1.0, 0.9)), ('cube_0', (1.8, 0.9))]:
    ok = rutil.pick(b, name)
    print(name, 'picked', ok, 'held at', np.round(b.obj(name), 3), 'tip', np.round(b.tip()[0], 3), 'steps', b.nsteps)
    final = rutil.place(b, name, dst)
    print(name, 'placed at', np.round(final, 3), 'target', dst, 'err', np.round(np.hypot(final[0]-dst[0], final[1]-dst[1]), 4), 'steps', b.nsteps)
