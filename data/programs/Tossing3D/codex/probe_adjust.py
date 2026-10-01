import numpy as np
from env_client import make_env


BASE_Q = np.array([.23, .39, 2.71, -.91, .17, -1.87, 2.46])


def values(state, obj, features):
    return np.array([float(state.get(obj, f)) for f in features])


def command(state, robot, base_target, q_target, grip):
    base = values(state, robot, ("pos_base_x", "pos_base_y", "pos_base_rot"))
    q = values(state, robot, tuple("pos_arm_joint%d" % i for i in range(1, 8)))
    be = np.asarray(base_target) - base
    be[2] = (be[2] + np.pi) % (2*np.pi) - np.pi
    qe = np.asarray(q_target) - q
    qe = (qe + np.pi) % (2*np.pi) - np.pi
    a = np.zeros(18, np.float32)
    a[:3] = np.clip(be, -.1, .1)
    a[3:10] = np.clip(qe, -.1, .1)
    a[10] = grip
    a[11:] = np.clip(5*qe, -3, 3)
    return a


def trial(dq2, dq4, close=1.0, offset=.45):
    env = make_env(); state, _ = env.reset(seed=0)
    cube = state.get_object_from_name("cube_0")
    robot = state.get_object_from_name("robot")
    p0 = values(state, cube, ("x", "y", "z"))
    base_target = np.array([p0[0] - offset, p0[1], 0.])
    q_target = BASE_Q.copy()
    q_target[1] += dq2
    q_target[3] += dq4
    q_target[5] -= dq4
    for _ in range(28):
        state, *_ = env.step(command(state, robot, base_target, q_target, 1-close))
    p_pose = values(state, cube, ("x", "y", "z"))
    for _ in range(8):
        state, *_ = env.step(command(state, robot, base_target, q_target, close))
    p_close = values(state, cube, ("x", "y", "z"))
    pull_target = base_target.copy(); pull_target[1] -= .40
    for _ in range(10):
        state, *_ = env.step(command(state, robot, pull_target, q_target, close))
    p_end = values(state, cube, ("x", "y", "z"))
    env.close()
    result = (np.linalg.norm(p_pose-p0), np.linalg.norm(p_close-p_pose),
              np.linalg.norm(p_end-p_close), p_end-p0)
    print("dq2 %.2f dq4 %.2f close %.0f pose %.4f closemove %.4f pull %.4f total %s" %
          (dq2, dq4, close, result[0], result[1], result[2],
           np.array2string(result[3], precision=3)), flush=True)
    return max(result[:3])


def reverse_current():
    """Test the closest rendered q6 candidate with reversed grip polarity."""
    env = make_env(); state, _ = env.reset(seed=0)
    cube = state.get_object_from_name("cube_0")
    robot = state.get_object_from_name("robot")
    p0 = values(state, cube, ("x", "y", "z"))
    grasp = np.array([.23, .39, 2.71, .20, .17, -1.57, 2.46])
    carry = np.array([0., -.15, np.pi, -1.35, 0., -1.55, np.pi/2])
    base = np.array([p0[0] - .42, p0[1], 0.])
    for _ in range(30):
        state, *_ = env.step(command(state, robot, base, grasp, 1.))
    p_pose = values(state, cube, ("x", "y", "z"))
    for _ in range(10):
        state, *_ = env.step(command(state, robot, base, grasp, 0.))
    p_close = values(state, cube, ("x", "y", "z"))
    for _ in range(18):
        state, *_ = env.step(command(state, robot, base, carry, 0.))
    p_lift = values(state, cube, ("x", "y", "z"))
    retreat = base.copy(); retreat[1] -= .4
    for _ in range(10):
        state, *_ = env.step(command(state, robot, retreat, carry, 0.))
    p_end = values(state, cube, ("x", "y", "z"))
    print("reverse current pose", p_pose-p0, "close", p_close-p_pose,
          "lift", p_lift-p_close, "retreat", p_end-p_lift,
          "total", p_end-p0, flush=True)
    env.close()


def wrist_roll_sweep():
    """Search wrist roll and then wrist yaw at the visually closest pose."""
    rolls = (0., .6, 1.2, 1.8, 2.4, 3.0, -.6)
    cases = [(q7, .17, close) for close in (1., 0.) for q7 in rolls]
    cases += [(2.46, q5, close) for close in (1., 0.)
              for q5 in (-.5, .67)]
    for q7, q5, close in cases:
        env = make_env(); state, _ = env.reset(seed=0)
        cube = state.get_object_from_name("cube_0")
        robot = state.get_object_from_name("robot")
        p0 = values(state, cube, ("x", "y", "z"))
        grasp = np.array([.23, .39, 3.05, 0., q5, -1.57, q7])
        carry = np.array([0., -.15, np.pi, -1.35, 0., -1.55, np.pi/2])
        base = np.array([p0[0] - .36, p0[1] - .12, 0.])
        for _ in range(28):
            state, *_ = env.step(command(state, robot, base, grasp, 1-close))
        p_pose = values(state, cube, ("x", "y", "z"))
        for _ in range(9):
            state, *_ = env.step(command(state, robot, base, grasp, close))
        p_close = values(state, cube, ("x", "y", "z"))
        for _ in range(16):
            state, *_ = env.step(command(state, robot, base, carry, close))
        p_lift = values(state, cube, ("x", "y", "z"))
        retreat = base.copy(); retreat[1] -= .35
        for _ in range(9):
            state, *_ = env.step(command(state, robot, retreat, carry, close))
        p_end = values(state, cube, ("x", "y", "z"))
        moves = [np.linalg.norm(p_pose-p0), np.linalg.norm(p_close-p_pose),
                 np.linalg.norm(p_lift-p_close), np.linalg.norm(p_end-p_lift)]
        print("q7 %.2f q5 %.2f close %.0f moves %s total %s" %
              (q7, q5, close, np.array2string(np.array(moves), precision=4),
               np.array2string(p_end-p0, precision=3)), flush=True)
        env.close()
        if max(moves) > .008:
            return


def main():
    cases = []
    for dq2 in (-.2, -.1, 0., .1):
        cases.append((dq2, 0.))
    for dq4 in (.25, .15, -.15, -.25):
        cases.append((0., dq4))
    for dq2 in (-.2, -.1):
        for dq4 in (.25, .15, -.15, -.25):
            cases.append((dq2, dq4))
    for case in cases:
        if trial(*case) > .008:
            return
    # Reverse polarity only for likely lower shoulder and both elbow directions.
    for case in ((-.2, .25), (-.2, -.25), (-.1, .15), (-.1, -.15)):
        if trial(*case, close=0.) > .008:
            return


if __name__ == "__main__":
    main()
