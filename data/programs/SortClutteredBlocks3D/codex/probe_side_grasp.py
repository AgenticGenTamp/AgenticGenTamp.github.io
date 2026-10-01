"""Try a force-closure side grasp at the calibrated narrow contact pose."""
from env_client import make_env
import numpy as np

env = make_env(); s, _ = env.reset(seed=0, options={"object_count": 4})
names = sorted(n for n in s.get_object_names() if n.startswith("cube"))
initial = {n: np.array([s.get(s.get_object_from_name(n), f)
                        for f in ("x", "y", "z")]) for n in names}
target = "cube4"; y = initial[target][1]
home = np.array([0., -.349, np.pi, -2.548, 0., -.873, np.pi / 2])
grasp = np.array([0., 1.3, np.pi, -1.7, 0., 1., np.pi / 2])

def move(base_goal, qgoal, grip, steps, speed=.1):
    global s
    for _ in range(steps):
        r = s.get_object_from_name("robot")
        b = np.array([s.get(r, f) for f in
                      ("pos_base_x", "pos_base_y", "pos_base_rot")])
        q = np.array([s.get(r, "pos_arm_joint%d" % i) for i in range(1, 8)])
        a = np.zeros(11, np.float32)
        a[:3] = np.clip(1.2 * (base_goal - b), -speed, speed)
        a[3:10] = np.clip(1.2 * (qgoal - q), -.1, .1)
        a[10] = grip
        s, reward, _, _, _ = env.step(a)

# Collision-free route to left; deploy with open jaws.
move(np.array([1., .6, np.pi]), home, 1., 30)
move(np.array([-1., .6, 0.]), home, 1., 35)
move(np.array([-1., y, 0.]), home, 1., 12)
move(np.array([-1., y, 0.]), grasp, 1., 85)
move(np.array([-.86, y, 0.]), grasp, 1., 18, .012)
move(np.array([-.86, y, 0.]), grasp, 0., 20)
print("closed", {n: np.round(np.array([s.get(s.get_object_from_name(n), f) for f in ("x","y","z")]), 3).tolist() for n in names})
# Lift mostly with shoulder, then retreat.
lift = grasp.copy(); lift[1] = .8
move(np.array([-.86, y, 0.]), lift, 0., 30)
move(np.array([-1.0, y, 0.]), lift, 0., 15)
print("lifted", {n: np.round(np.array([s.get(s.get_object_from_name(n), f) for f in ("x","y","z")]), 3).tolist() for n in names})
env.close()
