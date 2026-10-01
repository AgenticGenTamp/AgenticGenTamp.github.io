"""Small black-box probes for DynPushPullHook2DEnv controls."""

import sys
import numpy as np

from env_client import make_env


def val(state, obj, feat):
    return float(state.get(obj, feat))


def snapshot(state):
    out = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        row = {}
        for feat in ("x", "y", "theta", "held", "arm_joint", "arm_length", "finger_gap",
                     "base_radius", "gripper_base_width", "gripper_base_height",
                     "finger_height", "finger_width", "width", "height",
                     "length_side1", "length_side2"):
            try:
                row[feat] = val(state, obj, feat)
            except Exception:
                pass
        out[name] = row
    return out


def brief(state):
    snap = snapshot(state)
    keys = ["robot", "hook", "target_block"]
    keys += sorted(k for k in snap if k.startswith("obstruction"))
    return "\n".join(f"{k}: {snap[k]}" for k in keys if k in snap)


def run(seed, actions, every=1):
    env = make_env()
    state, info = env.reset(seed=seed)
    print(f"seed={seed} initial\n{brief(state)}")
    last = snapshot(state)
    for i, action in enumerate(actions, 1):
        state, reward, term, trunc, info = env.step(np.asarray(action, dtype=np.float64))
        cur = snapshot(state)
        changed_held = [(n, last[n].get("held"), cur[n].get("held")) for n in cur
                        if n in last and last[n].get("held") != cur[n].get("held")]
        if i % every == 0 or changed_held or term or trunc:
            rob = cur["robot"]
            hook = cur.get("hook", {})
            print(i, "a", action, "robot", {k: round(rob[k], 4) for k in rob},
                  "hook", {k: round(hook[k], 4) for k in hook},
                  "heldchange", changed_held, "done", term, trunc)
        last = cur
        if term or trunc:
            break
    env.close()


def grasp_once(seed, along, lateral=0.0, hook_along=0.0, align=False,
               angle_delta=0.0, preclose=False, push_steps=0):
    """Place the presumed gripper center on hook center and close."""
    env = make_env()
    state, _ = env.reset(seed=seed)
    robot = state.get_object_from_name("robot")
    hook = state.get_object_from_name("hook")
    theta = val(state, robot, "theta")
    hook_theta = val(state, hook, "theta")
    if preclose:
        # Narrow while far from the hook, retaining a little clearance over its width.
        for _ in range(9):
            state, _, _, _, _ = env.step(np.array([0, 0, 0, 0, -0.019]))
        robot = state.get_object_from_name("robot")
        hook = state.get_object_from_name("hook")
        theta = val(state, robot, "theta")
        hook_theta = val(state, hook, "theta")
    if align:
        desired_theta = hook_theta + angle_delta
        for _ in range(20):
            delta = (desired_theta - theta + np.pi) % (2 * np.pi) - np.pi
            da = np.clip(delta, -0.064, 0.064)
            state, _, _, _, _ = env.step(np.array([0, 0, da, -0.099, 0]))
            robot = state.get_object_from_name("robot")
            theta = val(state, robot, "theta")
            if abs(delta) < 1e-4:
                break
    hx = val(state, hook, "x") + hook_along * np.cos(hook_theta)
    hy = val(state, hook, "y") + hook_along * np.sin(hook_theta)
    goal_x = hx - along * np.cos(theta) + lateral * np.sin(theta)
    goal_y = hy - along * np.sin(theta) - lateral * np.cos(theta)
    # Move base, holding the open gripper and minimum arm extension fixed.
    for _ in range(100):
        robot = state.get_object_from_name("robot")
        dx = np.clip(goal_x - val(state, robot, "x"), -0.049, 0.049)
        dy = np.clip(goal_y - val(state, robot, "y"), -0.049, 0.049)
        state, _, term, trunc, _ = env.step(np.array([dx, dy, 0, -0.099, 0 if preclose else 0.019]))
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            break
    for _ in range(push_steps):
        # Keep the open gripper in contact while driving toward hook-local +x.
        state, _, term, trunc, _ = env.step(
            np.array([.049*np.cos(hook_theta), .049*np.sin(hook_theta), 0, 0, 0]))
    for i in range(18):
        state, _, term, trunc, _ = env.step(np.array([0, 0, 0, 0, -0.019]))
        hook = state.get_object_from_name("hook")
        if val(state, hook, "held"):
            break
    robot = state.get_object_from_name("robot")
    hook = state.get_object_from_name("hook")
    ans = (val(state, hook, "held"), val(state, robot, "finger_gap"),
           val(state, robot, "x"), val(state, robot, "y"),
           val(state, hook, "x"), val(state, hook, "y"))
    env.close()
    return ans


def held_motion(seed=0):
    """Known seed-0 grasp, translation, zero-action persistence, and release."""
    env = make_env()
    state, _ = env.reset(seed=seed)
    robot = state.get_object_from_name("robot")
    hook = state.get_object_from_name("hook")
    ht = val(state, hook, "theta")
    for _ in range(20):
        rt = val(state, robot, "theta")
        da = np.clip((ht - rt + np.pi) % (2*np.pi) - np.pi, -0.064, 0.064)
        state, _, _, _, _ = env.step(np.array([0, 0, da, -0.099, 0.019]))
        robot = state.get_object_from_name("robot")
        if abs(da) < 1e-4:
            break
    hook = state.get_object_from_name("hook")
    target_x = val(state, hook, "x") - 0.4*np.cos(ht)
    target_y = val(state, hook, "y") - 0.4*np.sin(ht)
    goal_x = target_x - .65*np.cos(ht)
    goal_y = target_y - .65*np.sin(ht)
    for _ in range(100):
        robot = state.get_object_from_name("robot")
        dx = np.clip(goal_x-val(state, robot, "x"), -.049, .049)
        dy = np.clip(goal_y-val(state, robot, "y"), -.049, .049)
        state, _, _, _, _ = env.step(np.array([dx, dy, 0, -.099, .019]))
        if abs(dx)+abs(dy) < 2e-4:
            break
    phases = [("close", [0, 0, 0, 0, -.019], 18),
              ("move", [.049, .02, 0, 0, -.019], 5),
              ("zero", [0, 0, 0, 0, 0], 5),
              ("open", [0, 0, 0, 0, .019], 12),
              ("away", [-.049, 0, 0, 0, .019], 5)]
    prior_held = val(state, state.get_object_from_name("hook"), "held")
    for label, action, count in phases:
        for index in range(count):
            state, _, _, _, _ = env.step(np.array(action))
            now_held = val(state, state.get_object_from_name("hook"), "held")
            if now_held != prior_held:
                print(label, "held transition at action", index + 1,
                      prior_held, "->", now_held)
            prior_held = now_held
        robot = state.get_object_from_name("robot")
        hook = state.get_object_from_name("hook")
        print(label, "robot", tuple(round(val(state,robot,k),3) for k in ("x","y","theta","finger_gap")),
              "hook", tuple(round(val(state,hook,k),3) for k in ("x","y","theta","held")))
    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "initial"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if mode == "initial":
        run(seed, [])
    elif mode == "axes":
        actions = [[0, 0, 0, 0, 0]] * 3
        for j in range(5):
            a = [0.0] * 5
            # Stay just inside bounds; server-side Box uses float32 endpoints.
            a[j] = [0.049, 0.049, 0.064, 0.099, 0.019][j]
            actions += [a] * 3 + [[-x for x in a]] * 3
        run(seed, actions)
    elif mode == "grasp":
        # Default seed 0: experimentally steer near hook, close, then translate.
        actions = [[0, 0, 0, 0, 0]] * 5
        actions += [[0, 0, 0, 0, 0.019]] * 10
        actions += [[0, 0, 0, 0.099, 0]] * 10
        actions += [[0, 0, 0, 0, -0.019]] * 12
        actions += [[0.049, 0, 0, 0, 0]] * 5
        run(seed, actions)
    elif mode == "scan_grasp":
        for lateral in (-0.20, -0.10, 0.0, 0.10, 0.20):
            for along in (0.45, 0.55, 0.65, 0.75, 0.85):
                print("along/lateral", along, lateral, "=>", grasp_once(seed, along, lateral))
    elif mode == "held_motion":
        held_motion(seed)
