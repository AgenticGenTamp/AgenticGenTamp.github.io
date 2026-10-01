"""Small black-box probes for PushPullHook2D robot kinematics."""
import argparse
import math
import numpy as np

from env_client import make_env


def short(s):
    return {
        "robot": np.round(s[0:9], 4).tolist(),
        "hook": np.round(s[9:20], 4).tolist(),
        "button": np.round(s[20:29], 4).tolist(),
        "target": np.round(s[29:38], 4).tolist(),
    }


def run(seed, actions):
    env = make_env()
    s, info = env.reset(seed=seed)
    print("initial", seed, short(s), "info", info)
    prior = s.copy()
    for i, action in enumerate(actions):
        s, reward, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        changed = [(j, round(float(s[j] - prior[j]), 5)) for j in range(38)
                   if abs(float(s[j] - prior[j])) > 1e-5]
        print(i, "a", action, "robot", np.round(s[:9], 4).tolist(),
              "objects", np.round(s[9:12], 4).tolist(), np.round(s[20:23], 4).tolist(),
              "changed", changed, "done", term, trunc)
        prior = s.copy()
        if term or trunc:
            break
    env.close()


def controls_to_pose(s, x, y, theta, arm=.1, vac=0.0):
    """Open-loop max-rate controls; observations make exact clipping unnecessary."""
    out = []
    # Retract, rotate, move, and finally extend to avoid sweeping contacts.
    out.append([0, 0, 0, -.1, vac])
    delta = (theta - float(s[2]) + math.pi) % (2 * math.pi) - math.pi
    while abs(delta) > 1e-5:
        da = max(-math.pi / 16, min(math.pi / 16, delta))
        out.append([0, 0, da, 0, vac]); delta -= da
    dx, dy = x - float(s[0]), y - float(s[1])
    while abs(dx) > 1e-5 or abs(dy) > 1e-5:
        ax, ay = max(-.05, min(.05, dx)), max(-.05, min(.05, dy))
        out.append([ax, ay, 0, 0, vac]); dx -= ax; dy -= ay
    if arm > .10001:
        out.append([0, 0, 0, arm - .1, vac])
    return out


def grasp_probe(seed, distance, arm):
    env = make_env(); s, _ = env.reset(seed=seed)
    bx, by = map(float, s[20:22])
    actions = controls_to_pose(s, bx - distance, by, 0.0, arm, 0.0)
    # Engage, then translate and rotate: attachment is visible in object deltas.
    actions += [[0, 0, 0, 0, 1], [.03, 0, 0, 0, 1],
                [0, .03, 0, 0, 1], [0, 0, .10, 0, 1],
                [0, 0, 0, 0, 0], [-.03, 0, 0, 0, 0]]
    last = s
    for i, a in enumerate(actions):
        s, _, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
        if i >= len(actions) - 6:
            print("probe", distance, arm, i - len(actions) + 6,
                  "r", np.round(s[:7], 4).tolist(),
                  "button", np.round(s[20:23], 4).tolist(),
                  "db", np.round(s[20:23] - last[20:23], 4).tolist())
        last = s
    env.close()


def hook_tip_probe(seed, arm, late_vac=False):
    """Seed-0 visual estimate: approach exposed hook tip from its right."""
    env = make_env(); s, _ = env.reset(seed=seed)
    # These are deliberately a safe low corridor, then an approach to seed-0 tip.
    v = 0.0 if late_vac else 1.0
    actions = controls_to_pose(s, 1.15, .46, math.pi, arm, v)
    actions += [[-.02, 0, 0, 0, v]] * 30
    if late_vac:
        actions += [[0, 0, 0, 0, 1]]
    actions += [[.03, 0, 0, 0, 1]] * 8
    if arm < .15:
        actions += [[0, 0, 0, .1, 1], [0, 0, 0, -.1, 1]]
    actions += [[0, .03, 0, 0, 1]] * 4
    actions += [[0, 0, .05, 0, 1]]
    actions += [[0, 0, 0, 0, 0], [.03, 0, 0, 0, 0]]
    last = s.copy()
    for i, a in enumerate(actions):
        s, _, _, _, _ = env.step(np.asarray(a, dtype=np.float32))
        dh = np.linalg.norm(s[9:12] - last[9:12])
        if dh > 1e-5 or i >= len(actions) - 45:
            print(i, "r", np.round(s[[0,1,2,4,6]], 4).tolist(),
                  "h", np.round(s[9:12], 4).tolist(), "dh", round(float(dh), 5))
        last = s.copy()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["init", "basis", "arm", "rotate", "grasp", "hook-tip"], default="basis")
    parser.add_argument("--distance", type=float, default=.4)
    parser.add_argument("--arm-value", type=float, default=.1)
    parser.add_argument("--late-vac", action="store_true")
    args = parser.parse_args()
    if args.mode == "grasp":
        grasp_probe(args.seed, args.distance, args.arm_value)
        raise SystemExit
    if args.mode == "hook-tip":
        hook_tip_probe(args.seed, args.arm_value, args.late_vac)
        raise SystemExit
    if args.mode == "init":
        actions = []
    elif args.mode == "basis":
        actions = [[.05, 0, 0, 0, 0], [0, .05, 0, 0, 0],
                   [0, 0, math.pi / 16, 0, 0], [0, 0, 0, .1, 0],
                   [0, 0, 0, 0, 1], [0, 0, 0, -.1, 1]]
    elif args.mode == "arm":
        actions = [[0, 0, 0, .1, 0]] * 8 + [[0, 0, 0, -.1, 0]] * 12
    else:
        actions = [[0, 0, math.pi / 16, 0, 0]] * 20
    run(args.seed, actions)
