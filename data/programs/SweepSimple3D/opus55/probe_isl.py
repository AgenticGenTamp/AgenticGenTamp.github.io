import sys, numpy as np
from env_client import make_env
from rutil import Bot
env = make_env(); obs, _ = env.reset(seed=1); b = Bot(env, obs)
for start, tgt in [((0.5, 1.8), (0.5, -1.0)), ((-1.0, 0.0), (1.0, 0.0)), ((0.2, -2.0), (0.2, 1.0)), ((1.6, 0.0), (0.0, 0.0))]:
    b.goto(base_t=[*start, 0.0], tol=0.02, max_steps=200)
    last = b.base(); st = 0
    for k in range(80):
        b.act(base_t=[*tgt, 0.0]); p = b.base()
        st = st + 1 if np.linalg.norm(p[:2] - last[:2]) < 0.003 else 0; last = p
        if st > 5: break
    print(start, '->', tgt, 'stop', np.round(b.base(), 3), flush=True)
