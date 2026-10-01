"""Controlled black-box probes for gripper release acceptance.

This is intentionally separate from approach.py.  Each trial resets the same
seed, grasps obstruction0 from the same pose, then varies only the requested
release pose and action sequence.
"""
from env_client import make_env
import math
import numpy as np


def val(state, name, feature):
    obj = state.get_object_from_name(name)
    return float(state.get(obj, feature))


def command(state, **goals):
    action = np.zeros(11, np.float32)
    mapping = {
        "base_x": (0, "pos_base_x"), "base_y": (1, "pos_base_y"),
        "q1": (3, "joint_1"), "q2": (4, "joint_2"),
        "q4": (6, "joint_4"), "q6": (8, "joint_6"),
    }
    for key, goal in goals.items():
        if key == "grip":
            action[10] = goal
        else:
            index, feature = mapping[key]
            action[index] = np.clip(goal - val(state, "robot", feature), -.2, .2)
    return action


def circle_center(src, dst, radius=.656):
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    distance = math.hypot(dx, dy)
    mid = ((src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2)
    height = math.sqrt(radius * radius - distance * distance / 4)
    choices = ((mid[0] - dy / distance * height, mid[1] + dx / distance * height),
               (mid[0] + dy / distance * height, mid[1] - dx / distance * height))
    return min(choices, key=lambda point: point[0])


def run_trial(final_q1, release_q6, sequence, seed=4):
    env = make_env()
    state, _ = env.reset(seed=seed)
    name = "obstruction0"
    src = (val(state, name, "pose_x"), val(state, name, "pose_y"))
    # This destination is used only to establish one reproducible grasp base.
    center = circle_center(src, (.2, 0.0))
    start_q1 = -math.atan2(src[1] - center[1], src[0] - center[0])
    for _ in range(20):
        state, _, _, _, _ = env.step(command(
            state, base_x=center[0], base_y=center[1], q1=start_q1,
            q2=.65, q4=-1.50, q6=-.87, grip=-1.0))
        if val(state, "robot", "grasp_active") > .5:
            break
    acquired = val(state, "robot", "grasp_active") > .5
    if acquired:
        # Standard carry lift, swing, then requested release height.
        for goal_q6 in (-.77, release_q6):
            for _ in range(10):
                state, _, _, _, _ = env.step(command(
                    state, q1=final_q1, q6=goal_q6))
                if (abs(val(state, "robot", "joint_1") - final_q1) < .005 and
                        abs(val(state, "robot", "joint_6") - goal_q6) < .005):
                    break
        before = tuple(val(state, name, f) for f in ("pose_x", "pose_y", "pose_z"))
        if sequence == "neutral":
            state, _, _, _, _ = env.step(command(state))
        elif sequence == "repeat_pose":
            state, _, _, _, _ = env.step(command(state, q1=final_q1, q6=release_q6))
        state, _, terminated, truncated, _ = env.step(command(state, grip=1.0))
        after = tuple(val(state, name, f) for f in ("pose_x", "pose_y", "pose_z"))
        held_after = val(state, "robot", "grasp_active") > .5
        finger = val(state, "robot", "finger_state")
    else:
        before = after = (float("nan"),) * 3
        held_after, finger, terminated, truncated = False, float("nan"), False, False
    env.close()
    return {
        "q1": final_q1, "q6": release_q6, "seq": sequence,
        "acquired": acquired, "before": tuple(round(x, 4) for x in before),
        "after": tuple(round(x, 4) for x in after), "held": held_after,
        "finger": round(finger, 4), "done": terminated, "trunc": truncated,
    }


if __name__ == "__main__":
    for q6 in (-.87, -.77, -.67):
        for q1 in (-1.2, -.6, 0.0, .6, 1.2):
            print(run_trial(q1, q6, "immediate"))
    for sequence in ("immediate", "neutral", "repeat_pose"):
        print(run_trial(1.2, -.87, sequence))
