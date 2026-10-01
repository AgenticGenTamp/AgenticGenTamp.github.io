"""Search empirical resets for a nearly tangent top-wall passage."""
import json
import time
from env_client import make_env
from approach import GeneratedApproach

env = make_env()
best = (-1.0, None)
started = time.monotonic()
for seed in range(10000, 15000):
    state, info = env.reset(seed=seed, options={'object_count': 8})
    heights = [float(state.get(obj, 'height'))
               for obj in state.get_objects(env.observation_space.get_type('rectangle'))
               if abs(float(state.get(obj, 'y'))) < 1e-8]
    height = max(heights, default=0.0)
    if height > best[0]:
        best = (height, seed)
        print('BEST', seed, height, 'elapsed', round(time.monotonic()-started, 2), flush=True)
    if height > 2.29985:
        break
print('SEARCH_RESULT', best, flush=True)
env.close()

env = make_env()
state, info = env.reset(seed=best[1], options={'object_count': 8})
policy = GeneratedApproach(env.action_space, env.observation_space, {})
policy.reset(state, info)
began = time.monotonic()
done = False
for step in range(400):
    action = policy.get_action(state)
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
robot = next(iter(state.get_objects(env.observation_space.get_type('crv_robot'))))
print('TEST_RESULT', json.dumps({'seed': best[1], 'height': best[0], 'done': done,
    'steps': step+1, 'seconds': time.monotonic()-began,
    'position': [float(state.get(robot, f)) for f in ('x', 'y', 'theta')],
    'path': [list(map(float, p)) for p in policy.path]}), flush=True)
env.close()
