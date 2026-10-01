"""Small black-box probes for DynPushPullHook2DEnv; not used by evaluation."""
import argparse
import numpy as np

from env_client import make_env


def val(s, obj, key):
    return float(s.get(obj, key))


def snapshot(s):
    rows = []
    for typ in ("kin_robot", "target_block", "hook", "dyn_rectangle"):
        for o in s.get_objects(typ):
            rows.append((typ, getattr(o, "name", str(o)), val(s, o, "x"), val(s, o, "y"),
                         val(s, o, "theta"), val(s, o, "held") if typ != "kin_robot" else 0))
    return rows


def toward(vec, speed=.049):
    n = max(float(np.linalg.norm(vec)), 1e-9)
    return np.clip(vec / n * min(speed, n), -speed, speed)


def angle_delta(want, have):
    return (want - have + np.pi) % (2*np.pi) - np.pi


def run(seed, strategy, limit=None):
    env = make_env()
    s, info = env.reset(seed=seed)
    lim = limit or env.max_steps
    initial = snapshot(s)
    target = s.get_object_from_name("target_block")
    hook = s.get_object_from_name("hook")
    robot = s.get_object_from_name("robot")
    ty0 = val(s, target, "y")
    hy0 = val(s, hook, "y")
    hx0 = val(s, hook, "x")
    ht0 = val(s, hook, "theta")
    trace = []
    term = trunc = False
    for t in range(lim):
        rx, ry = val(s, robot, "x"), val(s, robot, "y")
        tx, ty = val(s, target, "x"), val(s, target, "y")
        hx, hy = val(s, hook, "x"), val(s, hook, "y")
        rt = val(s, robot, "theta")
        if strategy == "up":
            a = [0, .049, 0, 0, 0]
        elif strategy == "arm":
            a = [0, 0, 0, .099, 0]
        elif strategy == "close_arm":
            a = [0, 0, 0, .099 if t < 15 else 0, -.019]
        elif strategy == "toward_target":
            # Steer base translationally toward target; keep arm extended.
            d = np.array([tx-rx, ty-ry])
            n = max(np.linalg.norm(d), 1e-9)
            a = [*(np.clip(d/n*.049, -.049, .049)), 0, .099, -.019]
        elif strategy == "toward_hook":
            d = np.array([hx-rx, hy-ry])
            n = max(np.linalg.norm(d), 1e-9)
            a = [*(np.clip(d/n*.049, -.049, .049)), 0, .099, -.019]
        elif strategy == "grasp_hook":
            # Retract/open, approach from the left, align and extend, then close.
            if t < 8:
                a = [0, 0, 0, -.099, .019]
            elif t < 70:
                waypoint = np.array([hx - .75, hy])
                dxy = toward(waypoint - np.array([rx, ry]))
                da = np.clip(angle_delta(0.0, rt), -.064, .064)
                a = [*dxy, da, -.099, .019]
            elif t < 85:
                da = np.clip(angle_delta(0.0, rt), -.064, .064)
                a = [0, 0, da, .099, .019]
            else:
                a = [0, 0, 0, 0, -.019]
        elif strategy == "grasp_hook2":
            # Put the default-length gripper around the hook from its left.
            waypoint = np.array([hx - .95, hy])
            if t < 55 and np.linalg.norm(waypoint - np.array([rx, ry])) > .025:
                dxy = toward(waypoint - np.array([rx, ry]))
                da = np.clip(angle_delta(0.0, rt), -.064, .064)
                a = [*dxy, da, 0, .019]
            else:
                a = [0, 0, np.clip(angle_delta(0.0, rt), -.064, .064), 0, -.019]
        elif strategy.startswith("grasp") and strategy[5:].isdigit():
            offset = int(strategy[5:]) / 100.0
            waypoint = np.array([hx0 - offset, hy0])
            dgoal = waypoint - np.array([rx, ry])
            if t < 55 and np.linalg.norm(dgoal) > .012:
                dxy = toward(dgoal)
                a = [*dxy, np.clip(angle_delta(0.0, rt), -.064, .064), 0, .019]
            else:
                a = [0, 0, np.clip(angle_delta(0.0, rt), -.064, .064), 0, -.019]
        elif strategy.startswith("grid_"):
            _, xs, ys = strategy.split("_")
            offset = int(xs) / 100.0
            yoff = int(ys) / 100.0
            waypoint = np.array([hx0 - offset, hy0 + yoff])
            dgoal = waypoint - np.array([rx, ry])
            if t < 45 and np.linalg.norm(dgoal) > .012:
                dxy = toward(dgoal)
                a = [*dxy, np.clip(angle_delta(0.0, rt), -.064, .064), 0, .019]
            elif t < 65:
                a = [0, 0, np.clip(angle_delta(0.0, rt), -.064, .064), 0, -.019]
            else:
                # A held hook should follow this downward-left test motion.
                a = [-.03, -.03, 0, 0, -.019]
        elif strategy in ("grasp_align", "grasp_solve"):
            direction = np.array([np.cos(ht0), np.sin(ht0)])
            waypoint = np.array([hx0, hy0]) - 1.43 * direction
            dgoal = waypoint - np.array([rx, ry])
            if t < 50 and (np.linalg.norm(dgoal) > .01 or abs(angle_delta(ht0, rt)) > .01):
                dxy = toward(dgoal)
                a = [*dxy, np.clip(angle_delta(ht0, rt), -.064, .064), 0, .019]
            elif t < 70:
                a = [0, 0, np.clip(angle_delta(ht0, rt), -.064, .064), 0, -.019]
            elif strategy == "grasp_align":
                dxy = -.03 * direction
                a = [*dxy, 0, 0, -.019]
            elif t < 115:
                # Swing the held long shaft vertical under the target column.
                dxy = toward(np.array([tx + .55, .65]) - np.array([rx, ry]), .035)
                a = [*dxy, np.clip(angle_delta(np.pi/2, rt), -.04, .04), 0, -.019]
            elif t < 155:
                a = [0, .025, 0, .05, -.019]
            elif t < 161:
                a = [-.03, 0, 0, 0, -.019]
            else:
                a = [0, -.035, 0, 0, -.019]
        elif strategy == "down":
            a = [0, -.049, 0, 0, 0]
        else:
            raise ValueError(strategy)
        s, rew, term, trunc, info = env.step(np.asarray(a, dtype=env.action_space.dtype))
        if t % 25 == 0:
            trace.append((t+1, val(s,target,"x"), val(s,target,"y"), val(s,hook,"x"), val(s,hook,"y"), val(s,hook,"held"),
                          val(s,robot,"x"), val(s,robot,"y"), val(s,robot,"theta"),
                          val(s,robot,"arm_length"), val(s,robot,"arm_joint"), val(s,robot,"finger_gap")))
        if term or trunc:
            break
    out = dict(seed=seed, strategy=strategy, steps=t+1, term=term, trunc=trunc,
               target_y0=ty0, target_y=val(s,target,"y"), hook_y0=hy0,
               hook_y=val(s,hook,"y"), initial=initial, trace=trace)
    env.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--strategies", default="up,arm,close_arm,toward_target,toward_hook,down")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    for seed in map(int, args.seeds.split(",")):
        for strategy in args.strategies.split(","):
            print(run(seed, strategy, args.limit))
