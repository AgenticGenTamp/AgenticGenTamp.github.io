"""Exercise target switching from an actual obstructed grasp configuration."""
from env_client import make_env
from approach import GeneratedApproach

e = make_env()
s, info = e.reset(seed=26, options={'object_count': 10})
p = GeneratedApproach(e.action_space, e.observation_space, {})
p.reset(s, info)
for _ in range(4):
    s, reward, terminated, truncated, info = e.step(p.get_action(s))
assert not s.get(p.r, 'grasp_active')
# Skip offset attempts to exercise retreat and selection of another real cube.
p.attempt_step = 12
for step in range(40):
    s, reward, terminated, truncated, info = e.step(p.get_action(s))
    if terminated or truncated:
        break
assert terminated, 'Target-switch recovery failed'
print('Target-switch recovery passed:', step + 1, 'actions, target', p.c.name)
e.close()
