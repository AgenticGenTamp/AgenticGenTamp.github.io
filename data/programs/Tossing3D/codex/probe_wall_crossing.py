"""Targeted test of the clearance needed to strike past the barrier end."""
import sys
import numpy as np
from env_client import make_env

FS = tuple("pos_arm_joint%d" % i for i in range(1, 8))
Q = np.array([.9441, -.0616, 1.4739, 1.3419, -.5038, -1.5938, .2053])
SWEEP = np.load("random_contact_found.npz")["actions"][139:145]


def val(state, name, features):
    obj = state.get_object_from_name(name)
    return np.array([float(state.get(obj, f)) for f in features])


def strike(env, state, alpha, scale, first):
    ca, sa = np.cos(alpha), np.sin(alpha)
    rot = np.array([[ca, -sa], [sa, ca]])
    offset = rot @ np.array([.6, .25])
    cube = val(state, "cube_0", ("x", "y"))
    for _ in range(55 if first else 32):
        base = val(state, "robot", ("pos_base_x", "pos_base_y"))
        yaw = val(state, "robot", ("pos_base_rot",))[0]
        joints = val(state, "robot", FS)
        action = np.zeros(18, np.float32)
        action[:2] = np.clip(cube - offset - base, -.06, .06)
        action[2] = np.clip(.4799 + alpha - yaw, -.06, .06)
        action[3:10] = np.clip(Q - joints, -.06, .06)
        action[10] = 1.
        action[11:] = np.clip(4 * (Q - joints), -4, 4)
        state, reward, term, trunc, _ = env.step(action)
    for raw in SWEEP:
        action = raw.copy()
        action[:2] = rot @ action[:2]
        action[10] = 1.
        action[11:] = np.clip(action[11:] * scale, -12, 12)
        state, reward, term, trunc, _ = env.step(action)
    for _ in range(3):
        action = np.zeros(18, np.float32)
        action[10] = 1.
        state, reward, term, trunc, _ = env.step(action)
    return state, term or trunc


clearance = float(sys.argv[1])
forward_alpha = float(sys.argv[2])
scale = float(sys.argv[3])
env = make_env()
s, _ = env.reset(seed=0, options={"object_count": 1})
crossed = None
for cycle in range(int(sys.argv[4]) if len(sys.argv) > 4 else 22):
    p0 = val(s, "cube_0", ("x", "y", "z"))
    alpha = -1.305 if p0[1] < clearance else forward_alpha
    # Preserve the calibrated weak lateral sweep while routing; ``scale`` is
    # deliberately only the post-clearance forward sweep strength.
    active_scale = .8 if p0[1] < clearance else scale
    s, done = strike(env, s, alpha, active_scale, cycle == 0)
    p1 = val(s, "cube_0", ("x", "y", "z"))
    if crossed is None and p1[0] > 1.34:
        crossed = (cycle, p1.copy())
    print(cycle, "a", alpha, "from", np.round(p0, 3), "to", np.round(p1, 3), flush=True)
    if done:
        break
print("RESULT", clearance, forward_alpha, scale, "crossed", crossed,
      "final", np.round(val(s, "cube_0", ("x", "y", "z")), 3), flush=True)
env.close()
