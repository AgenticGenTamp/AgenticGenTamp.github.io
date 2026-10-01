from env_client import make_env
from approach import GeneratedApproach
import json
import time
import sys
import math

class YawPolicy(GeneratedApproach):
    def reset(self, state, info):
        cubes = [o for o in state.get_objects(self.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube_')]
        self.cube_yaw = 2 * math.atan2(state.get(cubes[0], 'qz'), state.get(cubes[0], 'qw'))
        super().reset(state, info)
    def ik(self, x, z):
        q = super().ik(x, z)
        q[6] -= self.cube_yaw
        return q

for seed in ([2, 3, 4, 5, 7, 8] if '--yaw' in sys.argv else range(1, 9)):
    env = make_env()
    state, info = env.reset(seed=seed)
    cls = YawPolicy if '--yaw' in sys.argv else GeneratedApproach
    policy = cls(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    r = policy.robot
    initial = {
        'base': [state.get(r, f) for f in ('pos_base_x', 'pos_base_y', 'pos_base_rot')],
        'cubes': [[state.get(c, f) for f in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')] for c in policy.cubes],
    }
    start = time.time()
    for step in range(140):
        state, reward, terminated, truncated, info = env.step(policy.get_action(state))
        if terminated or truncated:
            break
    final = [[state.get(c, f) for f in ('x', 'y', 'z')] for c in policy.cubes]
    print(json.dumps({'seed': seed, 'initial': initial, 'final': final,
                      'grasp': any(p[2] > .4 for p in final), 'seconds': time.time()-start}), flush=True)
    env.close()
