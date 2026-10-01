"""Raster a rod beneath candidate arm poses; report exact grasp configuration."""
import sys
import numpy as np
from env_client import make_env


POSES = [
    [0, -1.57, 3.142, -1.57, 0, -1.57, 1.57],
    [0, 0, 3.142, -1.57, 0, -1.57, 1.57],
    [0, -1.57, 3.142, 0, 0, -1.57, 1.57],
    [0, 1.0, 3.142, -1.0, 0, -1.0, 1.57],
    [0, 0, 3.142, 0, 0, 0, 1.57],
    [0, -0.5, 3.142, 0, 0, 0, 1.57],
    [0, 0.5, 3.142, 0, 0, 0, 1.57],
    [0, -1.0, 3.142, 0, 0, 0, 1.57],
    [0, 1.57, 3.142, -2.5, 0, -1.0, 1.57],
    [0, -1.57, 3.142, 2.5, 0, 1.0, 1.57],
    [0, 1.2, 3.142, 2.2, 0, 1.2, 1.57],
    [0, -1.2, 3.142, -2.2, 0, -1.2, 1.57],
    [2.867, 1.088, -0.581, 1.371, -1.412, 2.09, 1.108],
    [-2.835, -1.999, -3.086, 1.829, -0.526, -2.09, 2.195],
    [2.23, 1.088, -0.581, -1.371, -1.412, 1.35, 1.108],
    [-2.23, -1.999, -3.086, -1.829, -0.526, -2.09, 2.195],
    [0.0, 0.0, np.pi, 0.5, 0.0, -0.3, np.pi/2],
    [1.395, 1.044, -2.626, -0.711, 1.519, -1.513, -0.648],
    [0.786, 0.309, -2.858, -0.526, 2.670, -1.611, -0.749],
    [-0.756, 0.914, -0.422, -1.25, 0.746, -2.09, -1.26],
    [2.23, 1.088, -0.581, -1.371, -1.412, 1.35, 1.108],
    [-2.23, -1.999, -3.086, -1.829, -0.526, -2.09, 2.195],
    [-2.121, 0.680, 0.862, -0.335, 2.027, 0.095, -1.375],
    [-2.064, 0.641, 0.967, -0.412, 1.980, 0.056, -0.664],
]
pose_index = int(sys.argv[1])
slice_index = int(sys.argv[2]) if len(sys.argv) > 2 else -1
env = make_env(); state, _ = env.reset(seed=1)
robot = state.get_object_from_name("robot")
rod = state.get_object_from_name("cuboid_1")


def g(obj, f): return float(state.get(obj, f))
def rodp(): return np.array([g(rod, f) for f in ("x", "y", "z")])
def advance(a):
    global state
    state, _, term, trunc, _ = env.step(np.asarray(a, np.float32))
    return term or trunc


orig = rodp(); target = np.asarray(POSES[pose_index])
initial_base = np.array([g(robot, "pos_base_x"), g(robot, "pos_base_y")])
for _ in range(180 if pose_index == 23 else 90):
    q = np.array([g(robot, f"pos_arm_joint{i}") for i in range(1, 8)])
    err = target - q
    # Gen3 joints 1/3/5/7 are continuous; 2/4/6 are bounded and must not
    # take a wrapped path to the opposite joint limit.
    err[[0, 2, 4, 6]] = (err[[0, 2, 4, 6]] + np.pi) % (2*np.pi) - np.pi
    a = np.zeros(11); a[3:10] = np.clip(.8*err, -.1, .1); a[10] = 1
    advance(a)
    if np.max(abs(err)) < .025: break
print("READY", pose_index, "q", np.round(q, 4), "rod", np.round(orig, 4), flush=True)

# Candidate poses are compact around the pedestal, so search base offsets +/- .38m.
span = .12 if pose_index == 23 else (.30 if pose_index == 22 else (.15 if pose_index == 16 else (.65 if pose_index >= 12 else .38)))
count = 17 if pose_index == 23 else (25 if pose_index == 22 else (16 if pose_index == 16 else (10 if pose_index in (20, 21) else (16 if pose_index >= 12 else 12))))
center_x = orig[0] - .68 if pose_index == 23 else (initial_base[0] if pose_index == 22 else (orig[0] - (.55 if pose_index == 16 else 0.0)))
center_y = orig[1] - .11 if pose_index == 23 else (initial_base[1] if pose_index == 22 else orig[1])
xs = center_x + np.linspace(-span, span, count)
ys = center_y + np.linspace(-span, span, count)
if pose_index in (22, 23) and slice_index in (0, 1, 2, 3):
    half = len(xs)//2
    xs = xs[:half+1] if slice_index % 2 == 0 else xs[half:]
    ys = ys[:half+1] if slice_index < 2 else ys[half:]
elif slice_index in (0, 1):
    ys = ys[slice_index * (len(ys)//2):(slice_index + 1) * (len(ys)//2)]
for iy, y in enumerate(ys):
    for x in (xs if iy % 2 == 0 else xs[::-1]):
        for _ in range(3):
            base = np.array([g(robot, "pos_base_x"), g(robot, "pos_base_y")])
            err = np.array([x, y]) - base
            if max(abs(err)) < .012: break
            a = np.zeros(11); a[:2] = np.clip(err/.87, -.1, .1); a[10] = 1
            advance(a)
        a = np.zeros(11); a[10] = 0
        repeats = 3 if pose_index == 23 else (1 if pose_index == 22 else (3 if slice_index >= 0 else (5 if pose_index in (19, 20) else 1)))
        for _ in range(repeats): advance(a)
        # A base jiggle discriminates an attachment from resting noise.
        a = np.zeros(11); a[0] = .025; a[10] = 0; advance(a)
        move = np.linalg.norm(rodp() - orig)
        if move > .004:
            base = [g(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")]
            q = [g(robot, f"pos_arm_joint{i}") for i in range(1, 8)]
            print("HIT pose", pose_index, "base", np.round(base, 5), "q", np.round(q, 5),
                  "rod", np.round(rodp(), 5), "delta", np.round(rodp()-orig, 5), flush=True)
            env.close(); raise SystemExit
        a = np.zeros(11); a[10] = 1; advance(a)
print("MISS", pose_index, flush=True)
env.close()
