"""Summarize one-cube approach phase transitions and geometry."""
from env_client import make_env
from approach import GeneratedApproach


def value(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


for seed in range(10):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    old_phase = None
    events = []
    terminated = False
    for step in range(500):
        action = policy.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        if policy.phase != old_phase:
            fields = []
            for name in ("box0", "cube0"):
                fields.append(name + "=" + ",".join(
                    f"{value(state, name, 'pose_' + c):.2f}" for c in "xyz"))
            fields.append("base=" + ",".join(
                f"{value(state, 'robot', f):.2f}"
                for f in ("pos_base_x", "pos_base_y")))
            fields.append(f"held={value(state, 'robot', 'grasp_active'):.0f}")
            events.append(f"{step}:{policy.phase} " + " ".join(fields))
            old_phase = policy.phase
        if terminated or truncated:
            break
    cube = [value(state, "cube0", "pose_" + c) for c in "xyz"]
    box = [value(state, "box0", "pose_" + c) for c in "xyz"]
    print(f"seed={seed} term={terminated} step={step} cube={cube} box={box}")
    print(" | ".join(events))
    env.close()
