"""Probe chassis endpoint nudges intended to rotate the wiper without tipping."""
import argparse
import math
import numpy as np
from env_client import make_env


def val(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def xy(state, name):
    return np.array([val(state, name, "x"), val(state, name, "y")])


def base(state):
    return np.array([val(state, "robot", "pos_base_x"), val(state, "robot", "pos_base_y")])


def attitude(state):
    qw, qx, qy, qz = [val(state, "wiper_0", f) for f in ("qw", "qx", "qy", "qz")]
    yaw = math.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
    # World z component of the wiper's local z axis: 1 is upright, 0 is on edge.
    upright = 1 - 2 * (qx*qx + qy*qy)
    return yaw, upright, (qw, qx, qy, qz)


def drive(env, state, target, speed, max_steps=100):
    path = []
    for _ in range(max_steps):
        delta = target - base(state)
        if np.linalg.norm(delta) < 0.025:
            break
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(0.65 * delta, -speed, speed)
        state, reward, term, trunc, _ = env.step(action)
        path.append(action[:2].copy())
        if term or trunc:
            break
    return state, path


def run(seed, endpoint, torque, push_speed, push_steps, stand_off):
    env = make_env()
    state, _ = env.reset(seed=seed, options={"object_count": 1})
    w0 = xy(state, "wiper_0")
    z0 = val(state, "wiper_0", "z")
    yaw0, up0, quat0 = attitude(state)
    half = val(state, "wiper_0", "bb_x")
    axis = np.array([math.cos(yaw0), math.sin(yaw0)])
    normal = np.array([-axis[1], axis[0]])
    # At +axis push +normal for CCW; reverse for clockwise.
    contact = w0 + endpoint * half * axis
    force = torque * endpoint * normal
    staging = contact - stand_off * force
    state, path = drive(env, state, staging, min(0.1, max(0.04, push_speed * 2)), 120)
    staged = base(state).copy()
    samples = []
    for step in range(push_steps):
        action = np.zeros(11, np.float32)
        action[:2] = push_speed * force
        state, reward, term, trunc, _ = env.step(action)
        if step in (0, 4, 9, 19, 39, push_steps - 1):
            yaw, up, quat = attitude(state)
            samples.append((step + 1, base(state).copy(), xy(state, "wiper_0").copy(),
                            val(state, "wiper_0", "z"), yaw, up, quat))
        if term or trunc:
            break
    yaw, up, quat = attitude(state)
    print("CONFIG", seed, endpoint, torque, push_speed, push_steps, stand_off)
    print("INITIAL w", np.round(w0, 5), "z", z0,
          "yaw", round(yaw0, 5), "upright", round(up0, 5), "quat", np.round(quat0, 5),
          "half", round(half, 5), "contact", np.round(contact, 5), "force", np.round(force, 5))
    print("STAGED base", np.round(staged, 5), "requested", np.round(staging, 5), "travel_steps", len(path))
    for sample in samples:
        step, rb, w, z, syaw, sup, squat = sample
        print("SAMPLE", step, "base", np.round(rb, 5), "w", np.round(w, 5), "dxy", np.round(w-w0, 5),
              "z", round(z, 5), "yaw", round(syaw, 5), "dyaw", round(syaw-yaw0, 5),
              "upright", round(sup, 5), "quat", np.round(squat, 5))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--endpoint", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--torque", type=int, choices=(-1, 1), default=-1,
                        help="-1 clockwise, +1 counterclockwise")
    parser.add_argument("--speed", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--stand-off", type=float, default=0.42)
    args = parser.parse_args()
    run(args.seed, args.endpoint, args.torque, args.speed, args.steps, args.stand_off)
