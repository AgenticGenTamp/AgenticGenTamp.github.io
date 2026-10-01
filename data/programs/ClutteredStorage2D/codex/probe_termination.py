"""One-block experiments for the shelf termination predicate.

This script deliberately imports env_client; approach.py remains host-safe.
"""

import sys

from env_client import make_env
from approach import GeneratedApproach


def f(state, obj, key):
    return float(state.get(obj, key))


def run(seed, x_offset=0.0, y_offset=0.055, retry_insert=False):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    shelf = state.get_objects(env.observation_space.get_type("shelf"))[0]
    slot_x = f(state, shelf, "x1") + f(state, shelf, "width1") / 2.0
    slot_y = f(state, shelf, "y1")
    print("START", seed, "slot",
          tuple(round(f(state, shelf, k), 5) for k in
                ("x1", "y1", "width1", "height1")))
    last_stage = None
    term = trunc = False
    for step in range(1000):
        action = policy.get_action(state)
        # Override only the placement coordinates; acquisition/orientation is
        # exactly the production policy. This permits controlled comparisons.
        policy.place = (slot_x + x_offset, slot_y + y_offset)
        policy.preplace = (slot_x + x_offset, slot_y - 0.035)
        if retry_insert and policy.stage == "release":
            policy.stage = "insert"
            policy.insert_stall = 0
            # Search the very narrow mouth while continuing to push inward.
            phase = (step // 2) % 5
            action[0] = (-0.012, -0.006, 0.0, 0.006, 0.012)[phase]
            action[3] = 0.025
            action[4] = 1.0
        state, reward, term, trunc, end_info = env.step(action)
        if policy.stage != last_stage or step % 100 == 99:
            robot = state.get_object_from_name("robot")
            block = state.get_object_from_name("block0")
            print(step + 1, policy.stage,
                  "r", tuple(round(f(state, robot, k), 4) for k in
                             ("x", "y", "theta", "arm_joint", "vacuum")),
                  "b", tuple(round(f(state, block, k), 4) for k in
                             ("x", "y", "theta")))
            last_stage = policy.stage
        if term or trunc:
            break
    robot = state.get_object_from_name("robot")
    block = state.get_object_from_name("block0")
    print("END", step + 1, term, trunc, end_info,
          "r", tuple(round(f(state, robot, k), 5) for k in
                     ("x", "y", "theta", "arm_joint", "vacuum")),
          "b", tuple(round(f(state, block, k), 5) for k in
                     ("x", "y", "theta")),
          "offsets", x_offset, y_offset, "retry", retry_insert)
    env.close()


if __name__ == "__main__":
    # seed [x_offset [y_offset [retry_insert]]]
    args = sys.argv[1:]
    if args:
        run(int(args[0]), *(float(x) for x in args[1:3]),
            retry_insert=(len(args) > 3 and bool(int(args[3]))))
    else:
        for default_seed in range(5):
            run(default_seed)
