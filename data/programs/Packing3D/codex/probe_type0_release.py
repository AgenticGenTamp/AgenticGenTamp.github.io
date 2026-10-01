"""Trace the current controller on one-object mirrored triangle episodes."""
from env_client import make_env
from approach import GeneratedApproach


def val(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def run(seed, limit=180):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space, env.make_primitives())
    policy.reset(state, info)
    print("seed", seed, "part", [round(val(state, "part0", x), 5) for x in ("pose_x", "pose_y", "pose_z")],
          "rack", [round(val(state, "rack", x), 5) for x in ("pose_x", "pose_y", "pose_z")],
          "type", val(state, "part0", "triangle_type"))
    old = None
    for step in range(limit):
        action = policy.get_action(state)
        phase = policy.phase
        state, reward, term, trunc, info = env.step(action)
        holding = val(state, "robot", "grasp_active") > .5
        now = (phase, holding)
        if now != old or step % 10 == 9 or term:
            print(step + 1, phase, "hold", holding,
                  "base", [round(val(state, "robot", x), 4) for x in ("pos_base_x", "pos_base_y", "pos_base_rot")],
                  "j2j4", [round(val(state, "robot", x), 4) for x in ("joint_2", "joint_4")],
                  "part", [round(val(state, "part0", x), 4) for x in ("pose_x", "pose_y", "pose_z")],
                  "a", [round(float(x), 3) for x in action], "term", term)
        old = now
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    for chosen_seed in (1, 6, 9):
        run(chosen_seed)
