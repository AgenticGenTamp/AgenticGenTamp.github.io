"""Focused shoulder/wrist grid near the visually aligned cube pose."""
from env_client import make_env
import numpy as np

env = make_env()
s, _ = env.reset(seed=0, options={"object_count": 4})
r = s.get_object_from_name("robot")
c = s.get_object_from_name("cube3")
start = np.array([s.get(c, f) for f in ("x", "y", "z")])

def command(state, base, q, grip, steps):
    robot = state.get_object_from_name("robot")
    for _ in range(steps):
        cur_b = np.array([state.get(robot, f) for f in
                          ("pos_base_x", "pos_base_y", "pos_base_rot")])
        cur_q = np.array([state.get(robot, "pos_arm_joint%d" % i)
                          for i in range(1, 8)])
        a = np.zeros(11, np.float32)
        a[:3] = np.clip(1.5 * (base - cur_b), -.1, .1)
        a[3:10] = np.clip(1.5 * (q - cur_q), -.1, .1)
        a[10] = grip
        state, _, _, _, _ = env.step(a)
    return state

base = np.array([.48, start[1], np.pi])
q = np.array([0., .9, np.pi, -1., 0., 1., np.pi / 2])
s = command(s, base, q, 0., 60)
starts = {}
for name in ("cube1", "cube2", "cube3", "cube4"):
    obj = s.get_object_from_name(name)
    starts[name] = np.array([s.get(obj, f) for f in ("x", "y", "z")])
for q2 in (.3, .8, 1.3, 1.8):
    for q4 in (-2.4, -1.7, -1.0, -.3):
        q[[1, 3, 5]] = (q2, q4, 1.0)
        # At each arm height, move the whole EE through the cube's world-x
        # coordinate; this removes uncertain camera depth from the search.
        s = command(s, np.array([1.0, start[1], np.pi]), q, 0., 25)
        s = command(s, np.array([.48, start[1], np.pi]), q, 0., 18)
        delta = 0.0
        for name, old in starts.items():
            obj = s.get_object_from_name(name)
            now = np.array([s.get(obj, f) for f in ("x", "y", "z")])
            delta = max(delta, float(np.linalg.norm(now - old)))
        print(q2, q4, round(delta, 4))
env.close()
