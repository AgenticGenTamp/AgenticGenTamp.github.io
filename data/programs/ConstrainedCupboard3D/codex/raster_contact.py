"""Fine base raster while holding a promising low end-effector pose."""
import sys
import numpy as np
from env_client import make_env

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
variant = int(sys.argv[2]) if len(sys.argv) > 2 else 0
mode = sys.argv[3] if len(sys.argv) > 3 else ""
branch = mode == "branch"
env = make_env(); state, _ = env.reset(seed=seed)
robot = state.get_object_from_name("robot")
names = sorted(n for n in state.get_object_names() if n.startswith("cuboid_"))
rod = state.get_object_from_name(names[0])
qgoal = np.array([1.571, -.819, 2.174, -2.059, -.974, -.682, -.677])
if variant:
    # Sweep the elbow/wrist bend about the IK seed to vary fingertip height.
    qgoal[3] += .3 * variant
if branch:
    qgoal = np.array([2.867, 1.088, -.581, -4.912, -1.412, -4.160, 1.108])
if mode == "corrected":
    qgoal = np.array([-3.1003, -2.4, .2638, -.4262, -1.5708, -.1238, 2.8392])
if mode == "refined":
    qgoal = np.array([-3.0175, -2.3524, -1.1879, -.314, -1.5966, .6707, -2.7101])


def v(obj, f): return float(state.get(obj, f))
def xyz(obj): return np.array([v(obj, f) for f in ("x", "y", "z")])
origin = xyz(rod)


def act(base_goal, grip, tag):
    global state
    a = np.zeros(11)
    nowb = np.array([v(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")])
    be = base_goal - nowb; be[2] = (be[2]+np.pi) % (2*np.pi)-np.pi
    a[:3] = np.clip(be/.87, -.1, .1)
    q = np.array([v(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
    qe = qgoal-q if branch else (qgoal-q+np.pi) % (2*np.pi)-np.pi
    a[3:10] = np.clip(.5*qe, -.1, .1); a[-1] = grip
    state, rew, term, trunc, _ = env.step(a.astype(np.float32))
    delta = xyz(rod)-origin
    if np.linalg.norm(delta) > .0025:
        base = [v(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")]
        qnow = [v(robot, f"pos_arm_joint{i}") for i in range(1,8)]
        print("HIT", tag, "delta", np.round(delta,6), "base", np.round(base,6),
              "q", np.round(qnow,6), "grip", v(robot,"pos_gripper"), flush=True)
        env.close(); raise SystemExit
    return max(np.max(np.abs(be)), np.max(np.abs(qe)))


# Reach pose safely at rear/side corner.
start = np.array([origin[0]-.75, origin[1]-.50, 0.])
for k in range(150): act(start, 1., ("settle",k))
# Serpentine 5 cm lattice spans plausible mount transforms. Keep closed while
# translating so even narrow fingertip contacts should register.
points=[]
for ix, dx in enumerate(np.arange(-.80, -.299, .10)):
    dys=list(np.arange(-.50, .501, .10))
    if ix%2: dys.reverse()
    points += [(origin[0]+dx, origin[1]+dy) for dy in dys]
for pi,(x,y) in enumerate(points):
    goal=np.array([x,y,0.])
    for k in range(10):
        act(goal, 0., ("raster",pi,k,"offset",round(x-origin[0],3),round(y-origin[1],3)))
qnow = np.array([v(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
base_now = np.array([v(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")])
print("NO_HIT", seed, "variant", variant, "target_q", qgoal,
      "final_q", np.round(qnow, 6), "base", np.round(base_now, 6),
      "delta", np.round(xyz(rod)-origin, 8), "points", len(points), flush=True)
env.close()
