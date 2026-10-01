"""Targeted seed-0 grasp grid; stop on the first object pulled with the base."""

import itertools
import numpy as np

from env_client import make_env


ROBOT = slice(16, 27)


def command_to(obs, base, joints, grip):
    target = np.r_[base, joints]
    error = target - obs[16:26]
    action = np.empty(11, dtype=np.float32)
    action[:10] = np.clip(error, -0.1, 0.1)
    action[10] = grip
    return action


def run_config(name, obj_i, dx, dy, q2, q4, q6):
    env = make_env()
    try:
        obs, _ = env.reset(seed=0)
        initial_obj = obs[obj_i:obj_i + 3].copy()
        base = np.array([initial_obj[0] - dx, initial_obj[1] - dy, 0.0])
        joints = np.array([0.0, q2, np.pi, q4, 0.0, q6, np.pi / 2])

        # Correct convention established visually: 0=open, 1=closed.
        steps = 0
        first_motion = None
        for _ in range(125):
            action = command_to(obs, base, joints, 0.0)
            obs, reward, term, trunc, _ = env.step(action)
            steps += 1
            if first_motion is None and np.linalg.norm(obs[obj_i:obj_i + 3] - initial_obj) > 0.002:
                first_motion = (steps, obs[16:27].copy(), obs[obj_i:obj_i + 3].copy())
            if np.max(np.abs(np.r_[base, joints] - obs[16:26])) < 0.006:
                break
        before_close = obs[obj_i:obj_i + 3].copy()
        reached = obs[16:26].copy()

        # Close while actively holding the approach pose.
        for _ in range(8):
            obs, reward, term, trunc, _ = env.step(command_to(obs, base, joints, 1.0))
            steps += 1
        after_close = obs[obj_i:obj_i + 3].copy()
        closed_pose = obs[16:27].copy()

        # Lift by reducing q2, as established from the rendered kinematics.
        lift_joints = joints.copy(); lift_joints[1] = 0.90
        path = []
        for k in range(100):
            lift_action = command_to(obs, base, lift_joints, 1.0)
            lift_action[4] = np.clip(lift_action[4], -0.025, 0.025)
            obs, reward, term, trunc, _ = env.step(lift_action)
            steps += 1
            if k % 5 == 0 or k == 99:
                path.append(obs[obj_i:obj_i + 3].copy())
            if abs(obs[20] - lift_joints[1]) < 0.012:
                break
        final_obj = obs[obj_i:obj_i + 3].copy()
        lifted = final_obj - after_close
        approach_delta = before_close - initial_obj
        contacted = np.linalg.norm(after_close - initial_obj) > 0.002
        success = final_obj[2] > initial_obj[2] + 0.035
        print("TRY", name, "dxdy", dx, dy, "q2q4q6", q2, q4, q6,
              "steps", steps, "reached_err", np.round(np.r_[base, joints] - reached, 4),
              "xyz0", np.round(initial_obj, 4),
              "approach_delta", np.round(approach_delta, 4),
              "contact_delta", np.round(after_close - before_close, 4),
              "closed_robot", np.round(closed_pose, 5),
              "lift_delta", np.round(lifted, 4), "final", np.round(final_obj, 4),
              "trajectory", np.round(np.asarray(path), 4).tolist(),
              "first_motion", None if first_motion is None else
              (first_motion[0], np.round(first_motion[1], 5).tolist(),
               np.round(first_motion[2], 5).tolist()),
              "SUCCESS", success, flush=True)
        return success
    finally:
        env.close()


def main():
    # Lower-wrist route, center first.
    poses = []
    for q2 in (1.65, 1.70):
        for dx in (0.49, 0.46, 0.52):
            for q6 in (-0.30, -0.35, -0.25):
                poses.append((dx, 0.01, q2, -1.46, q6))
    # Already tested: contacted and briefly lifted 7 mm, then slipped.
    poses.remove((0.49, 0.01, 1.65, -1.46, -0.30))
    objects = (("large", 0),)
    for obj_name, obj_i in objects:
        for cfg in poses:
            if run_config(obj_name, obj_i, *cfg):
                print("FIRST_SUCCESS", obj_name, cfg)
                return
    print("NO_SUCCESS")


if __name__ == "__main__":
    main()
