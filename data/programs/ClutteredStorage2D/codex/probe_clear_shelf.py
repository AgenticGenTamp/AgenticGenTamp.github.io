"""Focused seed-1 probe for clearing an occupied shelf (not evaluation code)."""

import math
import sys

import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def vals(state, obj, fields):
    return tuple(round(float(state.get(obj, f)), 4) for f in fields)


def run_contact_trial(base_x, theta, wiggle):
    env = make_env()
    state, info = env.reset(seed=1, options={"object_count": 3})
    robot = state.get_object_from_name("robot")
    block = state.get_object_from_name("block0")
    # Move below the pocket with a retracted arm, then extend into block 0.
    for _ in range(60):
        rx, ry, rt, arm = (float(state.get(robot, f)) for f in
                           ("x", "y", "theta", "arm_joint"))
        action = np.array([np.clip(base_x-rx, -.05, .05),
                           np.clip(2.10-ry, -.05, .05),
                           np.clip((theta-rt+math.pi) % (2*math.pi)-math.pi, -.196, .196),
                           np.clip(.2-arm, -.1, .1), 0.], dtype=np.float32)
        state, _, _, _, _ = env.step(action)
        if max(abs(base_x-rx), abs(2.10-ry)) < .008 and abs(action[2]) < .01:
            break
    for _ in range(8):
        arm = float(state.get(robot, "arm_joint"))
        state, _, _, _, _ = env.step(np.array([0, 0, 0, min(.1, .75-arm), 0], np.float32))
    before = vals(state, block, ("x", "y", "theta"))
    # Toggle vacuum, then slide contact as specified.
    state, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 1], np.float32))
    for _ in range(8):
        state, _, _, _, _ = env.step(np.array([wiggle[0], wiggle[1], wiggle[2],
                                                wiggle[3], 1], np.float32))
    after = vals(state, block, ("x", "y", "theta"))
    rafter = vals(state, robot, ("x", "y", "theta", "arm_joint", "vacuum"))
    env.close()
    return before, after, rafter


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "clear_then_policy":
        env = make_env()
        state, info = env.reset(seed=1, options={"object_count": 3})
        robot = state.get_object_from_name("robot")
        # Reproduce the successful grid contact: a slightly clockwise arm at
        # x=block0.x pushes the obstructing horizontal block to the back wall.
        for _ in range(60):
            rx, ry, rt, arm = (float(state.get(robot, f)) for f in
                               ("x", "y", "theta", "arm_joint"))
            et = (1.45-rt+math.pi) % (2*math.pi)-math.pi
            action = np.array([np.clip(4.4645-rx, -.05, .05),
                               np.clip(2.10-ry, -.05, .05), np.clip(et, -.196, .196),
                               np.clip(.2-arm, -.1, .1), 0.], np.float32)
            state, _, _, _, _ = env.step(action)
            if max(abs(4.4645-rx), abs(2.10-ry), abs(et)) < .01:
                break
        for _ in range(8):
            state, _, _, _, _ = env.step(np.array([0, 0, 0, .1, 0], np.float32))
        for _ in range(8):
            state, _, _, _, _ = env.step(np.array([0, .02, 0, 0, 1], np.float32))
        block0 = state.get_object_from_name("block0")
        print("after clear", vals(state, block0, ("x", "y", "theta")),
              vals(state, robot, ("x", "y", "theta", "arm_joint")))
        if len(sys.argv) > 2 and sys.argv[2] == "rotate":
            for _ in range(8):
                state, _, _, _, _ = env.step(np.array([0, 0, .196, 0, 1], np.float32))
                print("rotate", vals(state, block0, ("x", "y", "theta")),
                      vals(state, robot, ("x", "y", "theta", "arm_joint")))
            env.close()
            return
        if len(sys.argv) > 2 and sys.argv[2] in ("pull", "pull_policy"):
            for _ in range(20):
                state, _, _, _, _ = env.step(np.array([0, -.03, 0, 0, 1], np.float32))
                if sys.argv[2] == "pull":
                    print("pull", vals(state, block0, ("x", "y", "theta")),
                          vals(state, robot, ("x", "y", "theta", "arm_joint")))
            if sys.argv[2] == "pull":
                env.close()
                return
        # Explicitly detach, retract the tool while the base is fixed, then
        # withdraw straight down.  Moving on the vacuum-off edge still drags
        # the block for one simulation step.
        state, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 0], np.float32))
        for _ in range(4):
            state, _, _, _, _ = env.step(np.array([0, 0, 0, -.1, 0], np.float32))
        for _ in range(10):
            state, _, _, _, _ = env.step(np.array([0, -.05, 0, 0, 0], np.float32))
        print("after retreat", vals(state, robot, ("x", "y", "theta", "arm_joint")))
        policy = GeneratedApproach(env.action_space, env.observation_space,
                                   env.make_primitives())
        policy.reset(state, info)
        policy.to_clear = []
        if len(sys.argv) > 2 and sys.argv[2] not in ("pull_policy",):
            slot_x = float(sys.argv[2])
            policy._choose_slot = lambda _state, _exclude: (slot_x, 2.725)
        for step in range(900):
            if len(sys.argv) > 2 and sys.argv[2] == "pull_policy" and policy.stage == "insert":
                target = state.get_object_from_name(policy.target_name)
                if float(state.get(target, "y")) >= 2.70:
                    state, _, terminated, truncated, step_info = env.step(
                        np.array([0, 0, 0, 0, 0], np.float32))
                    policy.stage = "release"
            state, reward, terminated, truncated, step_info = env.step(policy.get_action(state))
            if step % 50 == 49:
                print(step + 1, policy.stage, policy.target_name,
                      [(b.name, vals(state, b, ("x", "y", "theta")))
                       for b in state.get_objects(policy.block_type)])
            if terminated or truncated:
                print("DONE", step + 1, terminated, truncated)
                break
        else:
            print("NOT DONE")
        env.close()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "contact_grid":
        for theta in (1.45, 1.52, 1.57, 1.62, 1.70):
            for xoff in (-.08, -.04, 0., .04, .08):
                result = run_contact_trial(4.4645+xoff, theta, (0, .02, 0, 0))
                if result[0] != result[1] or result[2][-1] != 1.0:
                    print(round(theta, 2), round(xoff, 2), result)
                else:
                    print(round(theta, 2), round(xoff, 2), "static", result[2])
        return
    env = make_env()
    state, info = env.reset(seed=1, options={"object_count": 3})
    robot = state.get_object_from_name("robot")
    shelf = state.get_object_from_name("shelf")
    blocks = list(state.get_objects(env.observation_space.get_type("target_block")))
    print("shelf", vals(state, shelf, ("x", "y", "theta", "width", "height",
                                         "x1", "y1", "theta1", "width1", "height1")))
    for block in blocks:
        print(block.name, vals(state, block, ("x", "y", "theta", "width", "height")))
    print("robot", vals(state, robot, ("x", "y", "theta", "base_radius", "arm_joint",
                                         "arm_length", "gripper_height", "gripper_width")))
    if len(sys.argv) > 1 and sys.argv[1] == "policy_clear":
        policy = GeneratedApproach(env.action_space, env.observation_space,
                                   env.make_primitives())
        policy.reset(state, info)
        policy.to_clear = ["block0"]
        for step in range(500):
            state, reward, terminated, truncated, step_info = env.step(policy.get_action(state))
            if step % 10 == 0 or policy.stage in ("clear_release", "choose"):
                block = state.get_object_from_name("block0")
                print(step + 1, policy.stage,
                      "r", vals(state, robot, ("x", "y", "theta", "arm_joint", "vacuum")),
                      "b0", vals(state, block, ("x", "y", "theta")))
            if terminated or truncated:
                print("DONE", step + 1, terminated, truncated)
                break
        print("final", [(b.name, vals(state, b, ("x", "y", "theta")))
                         for b in blocks])
    env.close()


if __name__ == "__main__":
    main()
