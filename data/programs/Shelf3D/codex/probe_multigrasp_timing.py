"""Trace stage transitions for the current policy on multi-cube episodes."""

from env_client import make_env
from approach import GeneratedApproach


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return tuple(round(float(state.get(obj, f)), 4) for f in ("x", "y", "z"))


def robot(state):
    obj = state.get_object_from_name("robot")
    vals = ("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper")
    return tuple(round(float(state.get(obj, f)), 4) for f in vals)


def run(seed=0, count=2, max_steps=1000):
    env = make_env()
    state, _ = env.reset(seed=seed, options={"object_count": count})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, {})
    old = (-1, -1, -1)
    try:
        for step in range(max_steps):
            marker = (policy.index, policy.stage, policy.retry)
            if marker != old:
                names = [n for n in state.get_object_names() if n.startswith("cube")]
                print(step, marker, robot(state), {n: xyz(state, n) for n in names})
                old = marker
            state, reward, term, trunc, _ = env.step(policy.get_action(state))
            if term or trunc:
                print("done", step + 1, reward, term, trunc,
                      {n: xyz(state, n) for n in names})
                break
    finally:
        env.close()


if __name__ == "__main__":
    run()
