"""Targeted seed-0 grasp sweep for the large cube; exploration only."""
import argparse
import itertools
import numpy as np

from env_client import make_env


OBJ = (0, 54, 70)


def control_to(env, obs, base_target, joint_target, grip, n):
    """Velocity-control the observed base/joints toward a configuration."""
    for _ in range(n):
        action = np.zeros(11, dtype=np.float32)
        base_err = np.asarray(base_target) - obs[16:19]
        action[:3] = np.clip(0.45 * base_err, -0.1, 0.1)
        joint_err = np.asarray(joint_target) - obs[19:26]
        # Joint velocity response is slower, so use saturated proportional drive.
        action[3:10] = np.clip(0.35 * joint_err, -0.1, 0.1)
        action[10] = grip
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    return obs, float(reward), bool(term or trunc)


def attempt(x_offset, y_offset, q2, open_cmd=0.0, close_cmd=1.0):
    env = make_env()
    obs, _ = env.reset(seed=0)
    initial = obs.copy()
    cube = initial[0:3].copy()
    base_target = (cube[0] - x_offset, cube[1] + y_offset, initial[18])
    joints = initial[19:26].copy()
    joints[1] = q2
    joints[3] = -1.5

    # Open before entering the near-ground configuration.
    obs, _, done = control_to(env, obs, initial[16:19], initial[19:26], open_cmd, 8)
    obs, _, done = control_to(env, obs, base_target, joints, open_cmd, 74)
    at_ground = obs.copy()
    obs, close_reward, done = control_to(env, obs, base_target, joints, close_cmd, 14)
    after_close = obs.copy()

    # Retract upward/backward while retaining the closed command.
    lift_joints = joints.copy()
    lift_joints[1] = 0.55
    lift_joints[3] = -2.15
    obs, lift_reward, done = control_to(env, obs, base_target, lift_joints, close_cmd, 42)
    env.close()

    starts = np.asarray([initial[i:i + 3] for i in OBJ])
    ground = np.asarray([at_ground[i:i + 3] for i in OBJ])
    closed = np.asarray([after_close[i:i + 3] for i in OBJ])
    lifted = np.asarray([obs[i:i + 3] for i in OBJ])
    dz = lifted[:, 2] - starts[:, 2]
    displacement = np.linalg.norm(lifted - starts, axis=1)
    success = bool(np.any(dz > 0.035))
    print("TRY", round(x_offset, 3), round(y_offset, 3), round(q2, 3),
          "base", np.round(at_ground[16:19], 3).tolist(),
          "j", np.round(at_ground[19:26], 3).tolist(),
          "grip", round(float(at_ground[26]), 3), "->", round(float(after_close[26]), 3),
          "obj_ground", np.round(ground - starts, 3).tolist(),
          "obj_close", np.round(closed - starts, 3).tolist(),
          "obj_lift", np.round(lifted - starts, 3).tolist(),
          "disp", np.round(displacement, 3).tolist(),
          "R", close_reward, lift_reward, "SUCCESS", success)
    return success


def deep_attempt(x_offset, q2, q4):
    """Approach corrected y alignment and stop immediately on object motion."""
    env = make_env()
    obs, _ = env.reset(seed=0)
    initial = obs.copy()
    cube = initial[:3].copy()
    base_target = np.array((cube[0] - x_offset, cube[1] - 0.05, initial[18]))
    joints = initial[19:26].copy()
    joints[1], joints[3] = q2, q4

    def phase(name, bt, jt, grip, steps):
        nonlocal obs
        for step in range(steps):
            a = np.zeros(11, np.float32)
            a[:3] = np.clip(.45 * (bt - obs[16:19]), -.1, .1)
            a[3:10] = np.clip(.35 * (jt - obs[19:26]), -.1, .1)
            a[10] = grip
            obs, r, term, trunc, _ = env.step(a)
            delta = np.asarray([obs[i:i + 3] - initial[i:i + 3] for i in OBJ])
            if np.max(np.linalg.norm(delta, axis=1)) > .002:
                print("MOTION", name, step + 1, "cmd", x_offset, q2, q4,
                      "base", np.round(obs[16:19], 4).tolist(),
                      "joints", np.round(obs[19:26], 4).tolist(),
                      "grip", round(float(obs[26]), 4),
                      "delta", np.round(delta, 4).tolist(), "reward", float(r))
                return True
            if term or trunc:
                return True
        return False

    found = phase("open", initial[16:19], initial[19:26], 0.0, 8)
    if not found:
        found = phase("descend", base_target, joints, 0.0, 94)
    reached = obs.copy()
    if not found:
        found = phase("close", base_target, joints, 1.0, 16)
    if not found:
        lift = joints.copy(); lift[1] = .55; lift[3] = -2.15
        found = phase("lift", base_target, lift, 1.0, 42)
    if not found:
        print("MISS cmd", x_offset, q2, q4,
              "actual_ground", np.round(reached[16:26], 4).tolist())
    env.close()
    return found


def small_attempt(x_offset, y_offset, q2=1.65, q6=-0.4):
    """Try SMALL1 with the visually promising vertical q6 posture."""
    env = make_env()
    obs, _ = env.reset(seed=0)
    initial = obs.copy()
    cube = initial[54:57].copy()
    base_target = np.array((cube[0] - x_offset, cube[1] + y_offset, initial[18]))
    joints = initial[19:26].copy()
    joints[1], joints[3], joints[5] = q2, -1.46, q6

    def phase(name, bt, jt, grip, steps):
        nonlocal obs
        for step in range(steps):
            action = np.zeros(11, np.float32)
            action[:3] = np.clip(.45 * (bt - obs[16:19]), -.1, .1)
            action[3:10] = np.clip(.35 * (jt - obs[19:26]), -.1, .1)
            action[10] = grip
            obs, reward, term, trunc, _ = env.step(action)
            delta = np.asarray([obs[i:i + 3] - initial[i:i + 3] for i in OBJ])
            if np.max(np.linalg.norm(delta, axis=1)) > .002:
                print("SMALL_MOTION", name, step + 1, "cmd", x_offset, y_offset, q2, q6,
                      "base", np.round(obs[16:19], 5).tolist(),
                      "joints", np.round(obs[19:26], 5).tolist(),
                      "grip", round(float(obs[26]), 5),
                      "delta", np.round(delta, 5).tolist(), "reward", float(reward))
                return True
            if term or trunc:
                return True
        return False

    found = phase("open", initial[16:19], initial[19:26], 0., 8)
    if not found:
        found = phase("descend", base_target, joints, 0., 96)
    reached = obs.copy()
    if not found:
        found = phase("close", base_target, joints, 1., 16)
    if not found:
        lift = joints.copy(); lift[1] = .55; lift[3] = -2.15
        found = phase("lift", base_target, lift, 1., 42)
    if not found:
        print("SMALL_MISS cmd", x_offset, y_offset, q2, q6,
              "actual_ground", np.round(reached[16:26], 5).tolist())
    env.close()
    return found


def sustained_small_attempt(x_offset):
    """Close around SMALL1, then lift slowly using shoulder q2 only."""
    env = make_env()
    obs, _ = env.reset(seed=0)
    initial = obs.copy()
    cube = initial[54:57].copy()
    base_target = np.array((cube[0] - x_offset, cube[1], initial[18]))
    joints = initial[19:26].copy()
    joints[1], joints[3], joints[5] = 1.7, -1.46, -.30

    # Open and settle at the grasp posture.
    obs, _, _ = control_to(env, obs, initial[16:19], initial[19:26], 0., 8)
    obs, _, _ = control_to(env, obs, base_target, joints, 0., 96)
    ground = obs.copy()
    obs, _, _ = control_to(env, obs, base_target, joints, 1., 12)
    closed = obs.copy()

    trace = []
    max_z = float(obs[56])
    first_up = None
    # Hold base and all other arm joints; apply a gentle negative q2 velocity.
    for step in range(50):
        action = np.zeros(11, np.float32)
        action[:3] = np.clip(.45 * (base_target - obs[16:19]), -.1, .1)
        err = joints - obs[19:26]
        action[3:10] = np.clip(.35 * err, -.1, .1)
        action[4] = -.025
        action[10] = 1.
        obs, reward, term, trunc, _ = env.step(action)
        max_z = max(max_z, float(obs[56]))
        if first_up is None and obs[56] > initial[56] + .008:
            first_up = (step + 1, obs.copy())
        if step % 5 == 4:
            trace.append(np.round(obs[54:57] - initial[54:57], 4).tolist())
        if term or trunc:
            break

    delta_ground = ground[54:57] - initial[54:57]
    delta_close = closed[54:57] - initial[54:57]
    delta_final = obs[54:57] - initial[54:57]
    sustained = bool(delta_final[2] > .008)
    print("SUSTAIN", x_offset, "base", np.round(ground[16:19], 5).tolist(),
          "ground_j", np.round(ground[19:26], 5).tolist(),
          "d_ground", np.round(delta_ground, 5).tolist(),
          "d_close", np.round(delta_close, 5).tolist(),
          "d_final", np.round(delta_final, 5).tolist(),
          "max_dz", round(max_z - float(initial[56]), 5),
          "trace", trace, "sustained", sustained,
          "first_up", None if first_up is None else
          [first_up[0], np.round(first_up[1][16:26], 5).tolist(),
           np.round(first_up[1][54:57] - initial[54:57], 5).tolist()])
    env.close()
    return sustained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=999)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--lower-small", action="store_true")
    parser.add_argument("--sustain-small", action="store_true")
    args = parser.parse_args()
    if args.sustain_small:
        for dx in (.42, .40, .44):
            if sustained_small_attempt(dx):
                return
        return
    if args.lower_small:
        configs = [(dx, 0., 1.7, q6) for q6 in (-.30, -.35, -.25)
                   for dx in (.49, .46, .52, .48)]
        for config in configs[args.start:args.stop]:
            if small_attempt(*config):
                return
        return
    if args.small:
        configs = [(dx, dy) for dy in (-.02, 0., .02) for dx in (.46, .43, .49, .45)]
        for config in configs[args.start:args.stop]:
            if small_attempt(*config):
                return
        return
    if args.deep:
        # Central geometry first; then bracket x. Within each, vary depth and wrist.
        configs = [(dx, q2, q4) for dx in (.42, .38, .46)
                   for q2, q4 in ((1.75, -1.45), (1.9, -1.45), (1.65, -1.45),
                                  (1.75, -1.6), (1.75, -1.3),
                                  (1.9, -1.6), (1.9, -1.3),
                                  (1.65, -1.6), (1.65, -1.3))]
        for config in configs[args.start:args.stop]:
            if deep_attempt(*config):
                return
        return
    # Centerline first, then small lateral corrections. Central x/q2 values first.
    centers = [(x, 0.0, q) for x, q in itertools.product(
        (0.44, 0.40, 0.48, 0.38, 0.50), (1.45, 1.30, 1.60))]
    laterals = [(x, y, q) for y in (-0.03, 0.03, -0.06, 0.06)
                for x in (0.44, 0.40, 0.48) for q in (1.45, 1.30, 1.60)]
    for config in (centers + laterals)[args.start:args.stop]:
        if attempt(*config):
            return

    # Polarity inversion was tested explicitly in the initial center sweep.


if __name__ == "__main__":
    main()
