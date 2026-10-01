"""Test lateral joint-1 motion after the policy's first narrow x push."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

env = make_env()
s, info = env.reset(seed=0, options={"object_count": 4})
agent = GeneratedApproach(env.action_space, env.observation_space, {})
agent.reset(s, info)
for _ in range(166):
    s, _, _, _, _ = env.step(agent.get_action(s))
robot = s.get_object_from_name("robot")
base_hold = np.array([s.get(robot, f) for f in
                      ("pos_base_x", "pos_base_y", "pos_base_rot")])
qgoal = np.array([0., 1.3, np.pi, -1.7, 0., 1., 0.])
for target_q1 in (-.05, -.10, -.15, -.20, -.25):
    qgoal[0] = target_q1
    for _ in range(8):
        robot = s.get_object_from_name("robot")
        base = np.array([s.get(robot, f) for f in
                         ("pos_base_x", "pos_base_y", "pos_base_rot")])
        q = np.array([s.get(robot, "pos_arm_joint%d" % i)
                      for i in range(1, 8)])
        a = np.zeros(11, np.float32)
        a[:3] = np.clip(1.5 * (base_hold - base), -.1, .1)
        a[3:10] = np.clip(1.5 * (qgoal - q), -.1, .1)
        a[10] = 0.
        s, reward, _, _, _ = env.step(a)
    positions = {}
    for name in sorted(n for n in s.get_object_names() if n.startswith("cube")):
        obj = s.get_object_from_name(name)
        positions[name] = tuple(round(s.get(obj, f), 3) for f in ("x", "y", "z"))
    print(target_q1, reward, positions)
env.close()
