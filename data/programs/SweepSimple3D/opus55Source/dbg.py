from run_test import *
import approach as A
s, info = env.reset(seed=0, options={'object_count': 1})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
for t in range(60):
    a = ap.get_action(s); s,*_ = env.step(a)
    b, arm, g, cubes, w = ap._read(s)
    tg = ap.queue[0] if ap.queue else None
    if tg is not None and tg.arm is not None:
        print(t, len(ap.queue), 'armerr', np.round(A.wrap(tg.arm-arm),3), 'baseerr', np.round(tg.base-b,3))
