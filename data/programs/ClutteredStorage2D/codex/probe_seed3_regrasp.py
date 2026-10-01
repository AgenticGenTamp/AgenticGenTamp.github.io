"""Probe recovery from seed-3/count-7 regrasp waypoint deadlock."""

import sys
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def val(state, obj):
    return tuple(round(float(state.get(obj, f)), 3) for f in ("x", "y", "theta"))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    env = make_env()
    state, info = env.reset(seed=3, options={"object_count": 7})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    pending_clear = []
    clear_started = False
    if mode in ("order_right", "order_right_down"):
        pending_clear = sorted(policy.to_clear,
                               key=lambda n: float(state.get(state.get_object_from_name(n), "x")),
                               reverse=True)
        policy.to_clear = pending_clear[:1]
    robot = state.get_object_from_name("robot")
    stalled = 0
    last = None
    detour = None
    forced_right = False
    for step in range(1500):
        if mode in ("order_right", "order_right_down"):
            if pending_clear and policy.clear_mode and policy.target_name == pending_clear[0]:
                clear_started = True
            if (clear_started and policy.stage == "choose" and policy.target_name is None
                    and policy.priority_name is None and not policy.to_clear):
                pending_clear.pop(0)
                clear_started = False
                if pending_clear:
                    policy.to_clear = pending_clear[:1]
        action = policy.get_action(state)
        if mode == "down_lower" and policy.stage == "orient" and policy.target_name:
            held = state.get_object_from_name(policy.target_name)
            if float(state.get(held, "y")) > 1.90:
                action = np.array([0, -.03, 0, 0, 1], np.float32)
        if mode == "right_grasp" and policy.stage == "insert" and policy.target_name:
            insert_target = state.get_object_from_name(policy.target_name)
            if (float(state.get(insert_target, "theta")) < -1.4 and
                    float(state.get(insert_target, "y")) >= 2.64):
                policy.stage = "release"
                action = np.array([0, 0, 0, 0, 0], np.float32)
        if mode == "right_grasp" and policy.stage == "retreat_regrasp" and not forced_right:
            target = state.get_object_from_name(policy.target_name)
            bx = float(state.get(target, "x")); by = float(state.get(target, "y"))
            policy.approach_theta = np.pi
            policy.goal_base = (bx + .54, by)
            policy.regrasp_safe = (bx + .54, max(.24, by - .50))
            policy.regrasp_side_x = bx + .50
            forced_right = True
        pos = (float(state.get(robot, "x")), float(state.get(robot, "y")))
        if policy.stage == "reposition_regrasp":
            stalled = stalled + 1 if last and max(abs(pos[i]-last[i]) for i in (0, 1)) < .001 else 0
            if stalled >= 3:
                if mode in ("down", "down_lower", "order_right_down"):
                    action = np.array([0, -.05, 0, 0, 0], np.float32)
                elif mode == "right":
                    action = np.array([.05, 0, 0, 0, 0], np.float32)
                elif mode == "rotate_down":
                    action = np.array([0, -.05, -.196, 0, 0], np.float32)
                elif mode == "detour":
                    detour = "right"
        else:
            stalled = 0
        if detour is not None:
            rx = float(state.get(robot, "x")); ry = float(state.get(robot, "y"))
            if detour == "right" and rx < 2.70:
                action = np.array([.05, 0, 0, 0, 0], np.float32)
            elif detour == "right":
                detour = "down"
                action = np.array([0, -.05, 0, 0, 0], np.float32)
            elif detour == "down" and ry > .75:
                action = np.array([0, -.05, 0, 0, 0], np.float32)
            elif detour == "down":
                detour = "across"
                action = np.array([-.05, 0, 0, 0, 0], np.float32)
            elif detour == "across" and rx > policy.regrasp_safe[0] + .01:
                action = np.array([-.05, 0, 0, 0, 0], np.float32)
            elif detour == "across":
                detour = "up"
                action = np.array([0, .05, 0, 0, 0], np.float32)
            elif detour == "up" and ry < policy.regrasp_safe[1] - .01:
                action = np.array([0, .05, 0, 0, 0], np.float32)
            else:
                detour = None
        last = pos
        state, reward, term, trunc, step_info = env.step(action)
        if step in (0, 80, 100, 120, 150) or (stalled == 3):
            target = state.get_object_from_name(policy.target_name) if policy.target_name else robot
            print(step+1, policy.stage, "robot", val(state, robot), "target", val(state, target),
                  "safe", getattr(policy, "regrasp_safe", None),
                  "side", getattr(policy, "regrasp_side_x", None),
                  "others", [(b.name, val(state, b)) for b in state.get_objects(policy.block_type)])
        if term or trunc:
            print("DONE", step+1, term)
            break
    else:
        target = state.get_object_from_name(policy.target_name) if policy.target_name else robot
        print("FAIL", policy.stage, val(state, robot), "target", val(state, target),
              "place", getattr(policy, "place", None))
    env.close()


if __name__ == "__main__":
    main()
