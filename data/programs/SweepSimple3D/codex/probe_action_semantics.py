"""Small black-box probes for SweepSimple3DEnv action semantics."""

import numpy as np

from env_client import make_env


def scalar(state, obj, feature):
    return float(state.get(obj, feature))


def robot_values(state):
    rob = state.get_object_from_name("robot")
    feats = [
        "pos_base_x", "pos_base_y", "pos_base_rot",
        *[f"pos_arm_joint{i}" for i in range(1, 8)],
        "pos_gripper",
        "vel_base_x", "vel_base_y", "vel_base_rot",
        *[f"vel_arm_joint{i}" for i in range(1, 8)],
        "vel_gripper",
    ]
    return {f: scalar(state, rob, f) for f in feats}


def movable_values(state):
    out = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        if obj.type.name != "mujoco_movable_object":
            continue
        out[obj.name] = np.array([scalar(state, obj, f) for f in ("x", "y", "z", "vx", "vy", "vz")])
    return out


def probe(index, value, steps=1, seed=0, env=None):
    owned = env is None
    if owned:
        env = make_env()
    state, info = env.reset(seed=seed)
    before = robot_values(state)
    objects_before = movable_values(state)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[index] = value
    total = 0.0
    terminated = truncated = False
    for _ in range(steps):
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
    after = robot_values(state)
    objects_after = movable_values(state)
    changed = {k: after[k] - before[k] for k in before if abs(after[k] - before[k]) > 1e-7}
    moved = {
        k: (objects_after[k] - objects_before[k]).round(6).tolist()
        for k in objects_before
        if np.max(np.abs(objects_after[k] - objects_before[k])) > 1e-6
    }
    print(f"idx={index} val={value:+.2f} n={steps} reward={total:+.4f} term={terminated}/{truncated}")
    print(" robot delta", {k: round(v, 6) for k, v in changed.items()})
    print(" object delta", moved)
    if owned:
        env.close()


if __name__ == "__main__":
    env = make_env()
    state, info = env.reset(seed=0)
    print("shape/bounds", env.action_space.shape, env.action_space.low.tolist(), env.action_space.high.tolist())
    print("max_steps", env.max_steps, "info", info)
    print("objects", state.get_object_names())
    print("robot initial", robot_values(state))
    print("movable initial", {k: v.tolist() for k, v in movable_values(state).items()})
    # Reuse one connection; reset restores exactly the same seed state.
    for idx in range(11):
        probe(idx, 0.1 if idx < 10 else 1.0, steps=1, env=env)
    for idx in range(10):
        probe(idx, -0.1, steps=10, env=env)
    probe(10, 1.0, steps=10, env=env)
    env.close()
