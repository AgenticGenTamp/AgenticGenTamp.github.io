"""Combine calibrated x base push with offset q1 lateral sweep."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import sys

env = make_env(); s, info = env.reset(seed=0, options={"object_count": 4})
agent = GeneratedApproach(env.action_space, env.observation_space, {})
agent.reset(s, info)
for _ in range(166): s, _, _, _, _ = env.step(agent.get_action(s))
cube = s.get_object_from_name("cube4")
target_y = s.get(cube, "y") + .042
r = s.get_object_from_name("robot")
radius = float(sys.argv[1]) if len(sys.argv) > 1 else .88
base_goal = np.array([s.get(cube, "x") - radius, target_y, 0.])
qgoal = np.array([0., 1.3, np.pi, -1.7, 0., 1., 0.])
for k in range(85):
    r = s.get_object_from_name("robot")
    b = np.array([s.get(r, f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
    q = np.array([s.get(r, "pos_arm_joint%d" % i) for i in range(1,8)])
    if k >= 15: qgoal[0] = .5
    a = np.zeros(11, np.float32)
    a[:3] = np.clip(1.2 * (base_goal-b), -.03, .03)
    a[3:10] = np.clip(1.2 * (qgoal-q), -.1, .1)
    a[10] = 0.
    s, reward, term, trunc, _ = env.step(a)
positions = {}
for n in sorted(n for n in s.get_object_names() if n.startswith("cube")):
    o=s.get_object_from_name(n); positions[n]=tuple(round(s.get(o,f),3) for f in ("x","y","z"))
print(radius, reward, term, positions)
env.close()
