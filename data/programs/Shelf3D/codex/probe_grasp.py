"""Search for a cube-contacting arm pose using object motion as feedback."""

import numpy as np

from env_client import make_env


JOINTS = [f"pos_arm_joint{i}" for i in range(1, 8)]


def read(state):
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube1")
    base = [state.get(robot, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")]
    joints = [state.get(robot, f) for f in JOINTS]
    grip = state.get(robot, "pos_gripper")
    xyz = [state.get(cube, f) for f in ("x", "y", "z")]
    return np.array(base), np.array(joints), float(grip), np.array(xyz)


def drive(env, state, action, steps):
    for _ in range(steps):
        state, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    return state


def main():
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    print("initial", [x.round(4).tolist() if hasattr(x, "round") else x for x in read(state)])
    for dim, val, n in ((0, .1, 10), (1, .1, 10), (2, .1, 10)):
        action = np.zeros(11, np.float32); action[dim] = val; action[10] = 1
        state = drive(env, state, action, n)
        print("after", dim, [x.round(4).tolist() if hasattr(x, "round") else round(x,4) for x in read(state)])
    env.close()

    # At the visually estimated gripper XY offset (+0.17 m base-x), sweep each
    # arm joint independently and report the first cube contact.
    for dim in range(3, 10):
        for sign in (-1.0, 1.0):
            env = make_env()
            state, _ = env.reset(seed=0, options={"object_count": 1})
            cube0 = read(state)[3].copy()
            # Put arm pedestal about 17 cm behind cube; global yaw is zero.
            for _ in range(20):
                base = read(state)[0]
                action = np.zeros(11, np.float32)
                action[0] = np.clip(2 * (cube0[0] - .17 - base[0]), -.1, .1)
                action[1] = np.clip(2 * (cube0[1] - base[1]), -.1, .1)
                state, _, _, _, _ = env.step(action)
            hit = None
            for k in range(45):
                action = np.zeros(11, np.float32)
                action[dim] = sign * .05
                state, _, term, trunc, _ = env.step(action)
                base, joints, _, cube = read(state)
                moved = float(np.linalg.norm(cube - cube0))
                if moved > .004:
                    hit = (k, joints.copy(), cube.copy(), moved)
                    break
                if term or trunc:
                    break
            if hit:
                print("HIT", dim, sign, hit[0], "q", hit[1].round(3).tolist(),
                      "cube", hit[2].round(3).tolist(), "moved", round(hit[3], 4))
            else:
                print("nohit", dim, sign, "qend", read(state)[1].round(3).tolist())
            env.close()

    # Sweep the default downward-facing gripper/base slowly toward cube1.
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube0 = read(state)[3].copy()
    for step in range(45):
        base, joints, grip, cube = read(state)
        action = np.zeros(11, np.float32)
        action[0] = .02
        action[1] = np.clip(2 * (cube0[1] - base[1]), -.03, .03)
        action[10] = 0.0
        state, reward, term, trunc, _ = env.step(action)
        base1, _, _, cube1 = read(state)
        delta = np.linalg.norm(cube1 - cube0)
        if step % 5 == 0 or delta > .002:
            print("sweep", step, "base", base1.round(3).tolist(),
                  "cube", cube1.round(3).tolist(), "moved", round(delta, 4))
        if delta > .02 or term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
