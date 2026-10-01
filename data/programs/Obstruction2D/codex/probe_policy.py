"""Standalone black-box probes for Obstruction2DEnv; never imported by approach.py."""

import argparse
import math
import numpy as np

from env_client import make_env


def objects(state, env):
    rows = []
    for typ in env.observation_space.types:
        for obj in state.get_objects(typ):
            row = {"name": obj.name, "type": typ.name}
            for feat in env.observation_space.type_features[typ]:
                row[feat] = float(state.get(obj, feat))
            rows.append(row)
    return rows


def dump(seed):
    env = make_env()
    state, info = env.reset(seed=seed)
    print("seed", seed, "max_steps", env.max_steps, "info", info)
    for row in objects(state, env):
        print(row)
    env.close()


def robot_and_named(state, env):
    rt = env.observation_space.get_type("crv_robot")
    r = state.get_objects(rt)[0]
    b = state.get_object_from_name("target_block")
    s = state.get_object_from_name("target_surface")
    return r, b, s


def grip_xy(state, r):
    x, y = state.get(r, "x"), state.get(r, "y")
    th = state.get(r, "theta")
    reach = state.get(r, "base_radius") + state.get(r, "arm_joint")
    return x + reach * math.cos(th), y + reach * math.sin(th)


def run_action(seed, action, n=20):
    env = make_env()
    state, _ = env.reset(seed=seed)
    r, b, s = robot_and_named(state, env)
    last = None
    for i in range(n):
        before = (state.get(r, "x"), state.get(r, "y"), state.get(r, "theta"),
                  state.get(r, "arm_joint"), state.get(r, "vacuum"),
                  state.get(b, "x"), state.get(b, "y"))
        state, rew, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        after = (state.get(r, "x"), state.get(r, "y"), state.get(r, "theta"),
                 state.get(r, "arm_joint"), state.get(r, "vacuum"),
                 state.get(b, "x"), state.get(b, "y"))
        if i in (0, n - 1) or after != last:
            print(i + 1, "before", tuple(round(x, 4) for x in before),
                  "after", tuple(round(x, 4) for x in after), "term", term,
                  "info", info, "grip", tuple(round(x, 4) for x in grip_xy(state, r)))
        last = after
        if term or trunc:
            break
    env.close()


def move_toward(env, state, target, vacuum, limit=100):
    """Translate/rotate/extend toward target robot config with clipped actions."""
    for _ in range(limit):
        r, _, _ = robot_and_named(state, env)
        cur = np.array([state.get(r, k) for k in ("x", "y", "theta", "arm_joint")])
        err = np.asarray(target) - cur
        if np.max(np.abs(err)) < 1e-4:
            return state, False
        act = np.array([np.clip(err[0], -.05, .05), np.clip(err[1], -.05, .05),
                        np.clip(err[2], -.196, .196), np.clip(err[3], -.1, .1), vacuum],
                       dtype=np.float32)
        state, _, term, trunc, _ = env.step(act)
        if term or trunc:
            return state, True
    return state, False


def probe_pick(seed, name, contact_offset=0.0):
    env = make_env()
    state, _ = env.reset(seed=seed)
    obj = state.get_object_from_name(name)
    ox, oy = state.get(obj, "x"), state.get(obj, "y")
    oh = state.get(obj, "height")
    # Full extension and downward-facing arm: nominal endpoint is base y - 0.3.
    # Approach at high y, then lower to the object's top.
    state, done = move_toward(env, state, (ox, .75, -math.pi / 2, .2), 0)
    state, done = move_toward(env, state,
                              (ox, oy + oh + .3 + contact_offset, -math.pi / 2, .2), 1)
    print("contact", name, "at", tuple(round(state.get(obj, k), 4) for k in ("x", "y")))
    state, done = move_toward(env, state, (ox, .75, -math.pi / 2, .2), 1)
    print("lift", name, "at", tuple(round(state.get(obj, k), 4) for k in ("x", "y")),
          "done", done)
    env.close()


def probe_sweep(seed, name, arm=.2):
    """Sweep the aligned robot downward with vacuum on and print object motion."""
    env = make_env()
    state, _ = env.reset(seed=seed)
    obj = state.get_object_from_name(name)
    ox = state.get(obj, "x")
    state, _ = move_toward(env, state, (ox, .75, -math.pi / 2, arm), 1)
    r, _, _ = robot_and_named(state, env)
    prev = None
    for i in range(16):
        state, _, term, trunc, _ = env.step(np.array([0, -.05, 0, 0, 1], np.float32))
        now = tuple(round(state.get(x, k), 4) for x, k in
                    ((r, "x"), (r, "y"), (obj, "x"), (obj, "y")))
        print(i, now, "moved_obj", prev is not None and now[2:] != prev[2:])
        prev = now
        if term or trunc:
            break
    # Vacuum attachment appears edge-triggered, so toggle it at established contact.
    state, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 0], np.float32))
    state, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 1], np.float32))
    for i in range(8):
        state, _, term, trunc, _ = env.step(np.array([0, .05, 0, 0, 1], np.float32))
        print("lift", i, tuple(round(state.get(x, k), 4) for x, k in
                               ((r, "x"), (r, "y"), (obj, "x"), (obj, "y"))))
        if term or trunc:
            break
    env.close()


def grid_pick(seed, name):
    """Search end-effector lateral offsets using fresh deterministic resets."""
    for xoff in np.linspace(-.12, .12, 13):
        env = make_env()
        state, _ = env.reset(seed=seed)
        obj = state.get_object_from_name(name)
        ox, oy = state.get(obj, "x"), state.get(obj, "y")
        state, _ = move_toward(env, state, (ox + xoff, .75, -math.pi / 2, .2), 0)
        for _ in range(10):
            state, _, term, trunc, _ = env.step(np.array([0, -.05, 0, 0, 1], np.float32))
        r, _, _ = robot_and_named(state, env)
        low_y = state.get(r, "y")
        for _ in range(8):
            state, _, term, trunc, _ = env.step(np.array([0, .05, 0, 0, 1], np.float32))
        moved = state.get(obj, "y") - oy
        print("offset", round(float(xoff), 3), "low_base_y", round(low_y, 4),
              "object_delta", round(moved, 4))
        env.close()


def probe_push(seed, name):
    env = make_env()
    state, _ = env.reset(seed=seed)
    obj = state.get_object_from_name(name)
    ox, oy = state.get(obj, "x"), state.get(obj, "y")
    # Point arm upward and put base immediately left at object mid-height.
    state, _ = move_toward(env, state, (ox - .2, oy + .1, math.pi / 2, .1), 0)
    r, _, _ = robot_and_named(state, env)
    for i in range(12):
        state, _, term, trunc, _ = env.step(np.array([.05, 0, 0, 0, 0], np.float32))
        print(i, "robot", tuple(round(state.get(r, k), 4) for k in ("x", "y")),
              "obj", tuple(round(state.get(obj, k), 4) for k in ("x", "y")))
        if term or trunc:
            break
    env.close()


def angled_pick(seed, name):
    """Search downward-diagonal approaches for low objects inaccessible from top."""
    for theta in np.linspace(-.2, -1.35, 8):
        env = make_env()
        state, _ = env.reset(seed=seed)
        obj = state.get_object_from_name(name)
        ox, oy = state.get(obj, "x"), state.get(obj, "y")
        h = state.get(obj, "height")
        cy = oy + h * .65
        # Empirical nominal gripper radius at full extension is about 0.3.
        bx = ox - .3 * math.cos(theta)
        by = cy - .3 * math.sin(theta)
        state, _ = move_toward(env, state, (bx, .75, theta, .2), 0)
        state, _ = move_toward(env, state, (bx, by, theta, .2), 0)
        # Advance along arm direction while energizing vacuum.
        for _ in range(5):
            state, _, _, _, _ = env.step(np.array([.01 * math.cos(theta),
                                                    .01 * math.sin(theta), 0, 0, 1], np.float32))
        r, _, _ = robot_and_named(state, env)
        low = (state.get(r, "x"), state.get(r, "y"))
        for _ in range(6):
            state, _, _, _, _ = env.step(np.array([0, .05, 0, 0, 1], np.float32))
        moved = state.get(obj, "y") - oy
        print("theta", round(float(theta), 3), "base", tuple(round(v, 3) for v in low),
              "delta", round(moved, 4))
        env.close()


def step_to(env, state, target, vacuum, maxn=80):
    """Like move_toward, but reports steps and termination."""
    used = 0
    for _ in range(maxn):
        r, _, _ = robot_and_named(state, env)
        cur = np.array([state.get(r, k) for k in ("x", "y", "theta", "arm_joint")])
        err = np.asarray(target) - cur
        if np.max(np.abs(err)) < 1e-4:
            break
        act = np.array([np.clip(err[0], -.05, .05), np.clip(err[1], -.05, .05),
                        np.clip(err[2], -.196, .196), np.clip(err[3], -.1, .1), vacuum],
                       dtype=np.float32)
        state, _, term, trunc, _ = env.step(act)
        used += 1
        if term or trunc:
            return state, used, True
    return state, used, False


def descend_to_contact(env, state, vacuum=1, maxn=20):
    r, _, _ = robot_and_named(state, env)
    used = 0
    delta = -.05
    for _ in range(maxn):
        old = state.get(r, "y")
        state, _, term, trunc, _ = env.step(np.array([0, delta, 0, 0, vacuum], np.float32))
        used += 1
        if term or trunc:
            return state, used, term or trunc
        if abs(state.get(r, "y") - old) < 1e-7:
            if delta == -.05:
                delta = -.01
            elif delta == -.01:
                delta = -.002
            else:
                return state, used, False
    return state, used, False


def approach_above(env, state, x):
    """Collision-safe unladen transit: stow arm, translate, then point down."""
    r, _, _ = robot_and_named(state, env)
    used = 0
    state, n, done = step_to(env, state,
                             (state.get(r, "x"), .88, 0.0, .1), 0); used += n
    state, n, done = step_to(env, state, (x, .88, 0.0, .1), 0); used += n
    state, n, done = step_to(env, state, (x, .88, -math.pi / 2, .2), 0); used += n
    return state, used, done


def solve_episode(seed, verbose=False):
    env = make_env()
    state, info = env.reset(seed=seed)
    rect_t = env.observation_space.get_type("rectangle")
    obstacles = [o for o in state.get_objects(rect_t) if o.name.startswith("obstruction")]
    obstacles.sort(key=lambda o: state.get(o, "y") + state.get(o, "height"), reverse=True)
    steps = 0
    block0 = state.get_object_from_name("target_block")
    surf0 = state.get_object_from_name("target_surface")
    bx0, bw0 = state.get(block0, "x"), state.get(block0, "width")
    sx0, sw0 = state.get(surf0, "x"), state.get(surf0, "width")
    # Use the farther world edge and lower objects before release. Suspended
    # releases remain suspended, and multiple objects otherwise block each other.
    task_mid = (min(bx0, sx0) + max(bx0 + bw0, sx0 + sw0)) / 2
    park_cx = .11 if task_mid >= .8 else 1.49
    # Remove all movable clutter to a common location far left of the work area.
    for obj in obstacles:
        cx = float(np.clip(state.get(obj, "x") + state.get(obj, "width") / 2,
                           .11, 1.49))
        state, n, done = approach_above(env, state, cx); steps += n
        state, n, done = descend_to_contact(env, state, 1); steps += n
        state, n, done = step_to(env, state, (cx, .85, -math.pi / 2, .2), 1); steps += n
        state, n, done = step_to(env, state, (park_cx, .85, -math.pi / 2, .2), 1); steps += n
        state, n, done = descend_to_contact(env, state, 1); steps += n
        state, _, done, trunc, _ = env.step(np.array([0, 0, 0, 0, 0], np.float32)); steps += 1
        state, n, done = step_to(env, state, (park_cx, .85, -math.pi / 2, .2), 0); steps += n
        if verbose:
            print("cleared", obj.name, "now", round(state.get(obj, "x"), 3),
                  round(state.get(obj, "y"), 3))
    block = state.get_object_from_name("target_block")
    surf = state.get_object_from_name("target_surface")
    true_bcx = state.get(block, "x") + state.get(block, "width") / 2
    bcx = float(np.clip(true_bcx, .11, 1.49))
    state, n, done = approach_above(env, state, bcx); steps += n
    state, n, done = descend_to_contact(env, state, 1); steps += n
    state, n, done = step_to(env, state, (bcx, .85, -math.pi / 2, .2), 1); steps += n
    # Preserve the off-center grasp transform near workspace boundaries.
    goal_cx = state.get(surf, "x") + state.get(surf, "width") / 2 + (bcx - true_bcx)
    state, n, done = step_to(env, state, (goal_cx, .85, -math.pi / 2, .2), 1); steps += n
    if not done:
        state, n, done = descend_to_contact(env, state, 1); steps += n
    if not done:
        state, _, done, trunc, _ = env.step(np.array([0, 0, 0, 0, 0], np.float32)); steps += 1
    if verbose:
        print("final block", *(round(state.get(block, k), 3) for k in ("x", "y")),
              "surface", *(round(state.get(surf, k), 3) for k in ("x", "y")))
    env.close()
    return done, steps, info["object_count"]


def benchmark(first, count):
    solved = 0
    vals = []
    for seed in range(first, first + count):
        ok, steps, nobj = solve_episode(seed)
        solved += ok
        vals.append(steps)
        print(seed, "objects", nobj, "solved", ok, "steps", steps)
    print("SUMMARY", solved, "/", count, "mean_steps", round(sum(vals) / len(vals), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dump", "action", "pick", "sweep", "grid", "push", "angled", "solve", "bench"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action", nargs=5, type=float,
                    default=[0, 0, 0, 0, 0])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--name", default="target_block")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--arm", type=float, default=.2)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()
    if args.mode == "dump":
        dump(args.seed)
    elif args.mode == "action":
        run_action(args.seed, args.action, args.steps)
    elif args.mode == "pick":
        probe_pick(args.seed, args.name, args.offset)
    elif args.mode == "sweep":
        probe_sweep(args.seed, args.name, args.arm)
    elif args.mode == "grid":
        grid_pick(args.seed, args.name)
    elif args.mode == "push":
        probe_push(args.seed, args.name)
    elif args.mode == "angled":
        angled_pick(args.seed, args.name)
    elif args.mode == "solve":
        print(solve_episode(args.seed, True))
    else:
        benchmark(args.seed, args.count)


if __name__ == "__main__":
    main()
