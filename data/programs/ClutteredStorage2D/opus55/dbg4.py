import sys
from env_client import make_env
from approach import GeneratedApproach
seed, oc, t0, t1 = map(int, sys.argv[1:5])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count': oc})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    if t >= t0:
        b = ap.blocks.get(ap.task[1]) if ap.task else None
        print(t, ap.task, ap.phase, a.round(3), round(ap.rx,3), round(ap.ry,3), round(ap.rth,3), round(ap.arm,3), ap.stuck, None if b is None else b['poly'][:,1].round(3).tolist())
    obs, r, term, trunc, info = env.step(a)
