"""Broad deterministic-random search for any arm-to-rod contact."""
import sys
import numpy as np
from env_client import make_env


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
trial_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 8123
env = make_env()
state, _ = env.reset(seed=seed)
robot = state.get_object_from_name("robot")
names = sorted(n for n in state.get_object_names() if n.startswith("cuboid_"))
rod = state.get_object_from_name(names[0])


def val(obj, feature):
    return float(state.get(obj, feature))


def xyz(obj):
    return np.array([val(obj, f) for f in ("x", "y", "z")])


origin = xyz(rod)
print("START", seed, names[0], np.round(origin, 5), flush=True)


def advance(action, tag):
    global state
    state, reward, term, trunc, _ = env.step(np.asarray(action, np.float32))
    delta = xyz(rod) - origin
    if np.linalg.norm(delta) > .003:
        base = [val(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")]
        q = [val(robot, f"pos_arm_joint{i}") for i in range(1, 8)]
        print("HIT", tag, "delta", np.round(delta, 6), "base", np.round(base, 6),
              "q", np.round(q, 6), "grip", val(robot, "pos_gripper"), flush=True)
        env.close()
        raise SystemExit(0)


# Put the base beside the chosen rod. Across trials vary both its offset and
# heading; random joint targets are reached with bounded feedback afterwards.
rng = np.random.default_rng(trial_seed)
offset_x = rng.uniform(-.85, -.35)
offset_y = rng.uniform(-.45, .45)
yaw = rng.uniform(-np.pi, np.pi)
goal = np.array([origin[0] + offset_x, origin[1] + offset_y, yaw])
for k in range(80):
    now = np.array([val(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")])
    err = goal - now
    err[2] = (err[2] + np.pi) % (2*np.pi) - np.pi
    a = np.zeros(11); a[:3] = np.clip(err / .87, -.1, .1); a[-1] = 1.
    advance(a, ("base", k, np.round(goal, 4).tolist()))
    if max(abs(err)) < .015:
        break

# Correlated random walk gives many continuous swept volumes, unlike isolated
# random poses. Occasionally reverse the gripper while retaining arm motion.
direction = rng.choice([-1., 1.], 7)
for k in range(800):
    if k % 12 == 0:
        direction = rng.choice([-1., 1.], 7)
    if k % 47 == 0:
        direction[rng.integers(7)] *= -1
    a = np.zeros(11)
    a[3:10] = .1 * direction
    a[-1] = 1. if (k // 20) % 2 == 0 else -1.
    advance(a, ("arm", k, trial_seed))

print("NO_HIT", seed, trial_seed, "goal", np.round(goal, 5), flush=True)
env.close()
