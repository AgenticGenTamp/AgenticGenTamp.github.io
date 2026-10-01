"""Probe the exact box-only table success condition.

This is deliberately standalone exploration code and is never imported by
approach.py.
"""
import sys

import numpy as np

from env_client import make_env


Q = [
    [.951305288, 1.970876128, -1.493305392, -1.219847079,
     4.230377134, .474353059, 6.00596593],
    [4.155821175, .426711007, -1.502109215, -1.900626345,
     1.884001343, .06090929, 5.236141916],
    [3.870779777, 1.703020948, -2.855671258, -.627929854,
     4.26973606, 1.31947716, 7.283258434],
]
OFF = [[-.799987478, -.347800459, 2.104792961],
       [.126212298, -.083534270, -3.139703362],
       [-.428185141, .054943428, 2.978131848]]


def v(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def act_toward(s, base, q, grip):
    a = np.zeros(11, np.float32)
    current_base = [v(s, "robot", "pos_base_x"),
                    v(s, "robot", "pos_base_y"),
                    v(s, "robot", "pos_base_rot")]
    current_q = [v(s, "robot", "joint_%d" % i) for i in range(1, 8)]
    a[:3] = np.clip(np.asarray(base) - current_base, -.2, .2)
    a[3:10] = np.clip(np.asarray(q) - current_q, -.2, .2)
    a[10] = grip
    return a


def run(target_x=.6, target_y=0., lower_steps=12, seed=0, scan=False,
        descend=0, final_delta=.0):
    e = make_env()
    s, info = e.reset(seed=seed, options={"object_count": 0})
    tx, ty = v(s, "box0", "pose_x"), v(s, "box0", "pose_y")
    terminated = False
    # The empirically discovered collision-aware grasp route.
    for route_index, (q0, off) in enumerate(zip(Q, OFF)):
        q = np.array(q0)
        if route_index == 2:
            q[0] += final_delta
        base = [tx + off[0], ty + off[1], off[2]]
        for _ in range(35):
            s, _, terminated, _, _ = e.step(act_toward(s, base, q, 1.))
        for _ in range(2):
            s, _, terminated, _, _ = e.step(act_toward(s, base, q, -1.))
    print("grasp", v(s, "robot", "grasp_active"),
          [round(v(s, "box0", "pose_" + c), 4) for c in "xyz"])

    grasp_q = np.array([v(s, "robot", "joint_%d" % i)
                        for i in range(1, 8)])
    lift_q = grasp_q.copy()
    lift_q[1] -= 1.6
    # Lift while holding base stationary.
    base = [v(s, "robot", "pos_base_x"), v(s, "robot", "pos_base_y"),
            v(s, "robot", "pos_base_rot")]
    for _ in range(10):
        s, _, terminated, _, _ = e.step(act_toward(s, base, lift_q, -1.))

    # Translate using object feedback; preserve the lifted arm pose.
    for _ in range(20):
        bx, by = v(s, "robot", "pos_base_x"), v(s, "robot", "pos_base_y")
        base = [bx + target_x - v(s, "box0", "pose_x"),
                by + target_y - v(s, "box0", "pose_y"),
                v(s, "robot", "pos_base_rot")]
        s, _, terminated, _, _ = e.step(act_toward(s, base, lift_q, -1.))

    # Lower while counter-translating the base to cancel the arm's horizontal
    # arc and keep the box at the requested point.
    for k in range(lower_steps):
        bx, by = v(s, "robot", "pos_base_x"), v(s, "robot", "pos_base_y")
        base = [bx + target_x - v(s, "box0", "pose_x"),
                by + target_y - v(s, "box0", "pose_y"),
                v(s, "robot", "pos_base_rot")]
        s, _, terminated, _, _ = e.step(act_toward(s, base, grasp_q, -1.))
        print("lower", k, [round(v(s, "box0", "pose_" + c), 4)
                           for c in "xyz"])

    if scan:
        # Reversible local sensitivity probe at the blocked placement pose.
        for j in range(7):
            for direction in (-1., 1.):
                before = np.array([v(s, "box0", "pose_" + c) for c in "xyz"])
                a = np.zeros(11, np.float32); a[3 + j] = .1 * direction; a[10] = -1.
                s = e.step(a)[0]
                after = np.array([v(s, "box0", "pose_" + c) for c in "xyz"])
                # Undo accepted motion before probing the next coordinate.
                a[3 + j] *= -1
                s = e.step(a)[0]
                print("scan", j + 1, int(direction), "delta",
                      np.round(after - before, 5).tolist())

    # Joint 4 negative is the locally measured descent direction at the
    # q2/table-blocked pose. Counter-translate to retain horizontal centering.
    for k in range(descend):
        # Move q2 away from its collision boundary, then trade that gained
        # clearance for twice as much negative-q4 motion.
        for action_index, amount in ((4, -.1), (6, -.2)):
            a = np.zeros(11, np.float32)
            a[0] = np.clip(target_x - v(s, "box0", "pose_x"), -.2, .2)
            a[1] = np.clip(target_y - v(s, "box0", "pose_y"), -.2, .2)
            a[action_index] = amount
            a[10] = -1.
            s = e.step(a)[0]
        print("descend", k, [round(v(s, "box0", "pose_" + c), 4)
                             for c in "xyz"])

    a = np.zeros(11, np.float32)
    a[10] = 1.
    print("prerelease", "base",
          [round(v(s, "robot", f), 7) for f in
           ("pos_base_x", "pos_base_y", "pos_base_rot")],
          "q", [round(v(s, "robot", "joint_%d" % i), 7)
                for i in range(1, 8)],
          "boxquat", [round(v(s, "box0", "pose_q" + c), 7)
                      for c in "xyzw"])
    for k in range(6):
        s, reward, terminated, truncated, _ = e.step(a)
        print("release", k, reward, terminated, truncated,
              [round(v(s, "box0", "pose_" + c), 4) for c in "xyz"],
              "held", v(s, "robot", "grasp_active"))
        if terminated or truncated:
            break
    e.close()
    return terminated


def search_higher_grasp():
    """Vary final q1 and report attachment height; negative tf-z is ideal."""
    e = make_env()
    for delta in np.arange(.1, 1.01, .1):
        s, _ = e.reset(seed=0, options={"object_count": 0})
        tx, ty = v(s, "box0", "pose_x"), v(s, "box0", "pose_y")
        for index in range(2):
            base = [tx + OFF[index][0], ty + OFF[index][1], OFF[index][2]]
            for _ in range(35):
                s = e.step(act_toward(s, base, Q[index], 1.))[0]
        q = np.array(Q[2]); q[0] += delta
        base = [tx + OFF[2][0], ty + OFF[2][1], OFF[2][2]]
        for _ in range(35): s = e.step(act_toward(s, base, q, 1.))[0]
        for _ in range(2): s = e.step(act_toward(s, base, q, -1.))[0]
        print("high", round(delta, 2), "held", v(s, "robot", "grasp_active"),
              "tf", [round(v(s, "robot", "grasp_tf_" + c), 4)
                     for c in "xyz"])
    e.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "high":
        search_higher_grasp()
        raise SystemExit
    x = float(sys.argv[1]) if len(sys.argv) > 1 else .6
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    mode = sys.argv[4] if len(sys.argv) > 4 else ""
    descend = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    final_delta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.
    run(x, y, steps, scan=(mode == "scan"), descend=descend,
        final_delta=final_delta)
