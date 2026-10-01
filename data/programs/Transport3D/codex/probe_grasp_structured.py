"""Structured Franka-kinematics grasp probe; never imported by approach.py."""
import json
import math

import numpy as np
from scipy.optimize import least_squares

from env_client import make_env


JOINT_OFFSET = np.array([0.0, 0.0, math.pi, 0.0, 0.0, math.pi,
                         -math.pi / 4.0])


def val(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def fk(q):
    """Conventional Franka DH hypothesis in its arm-base coordinate frame."""
    aa = [0, 0, 0, .0825, -.0825, 0, .088]
    dd = [.333, 0, .316, 0, .384, 0, .107]
    al = [0, -math.pi/2, math.pi/2, math.pi/2,
          -math.pi/2, math.pi/2, math.pi/2]
    t = np.eye(4)
    for qi, a, d, alpha in zip(q, aa, dd, al):
        c, s, ca, sa = math.cos(qi), math.sin(qi), math.cos(alpha), math.sin(alpha)
        t = t @ np.array([[c, -s*ca, s*sa, a*c],
                          [s, c*ca, -c*sa, a*s],
                          [0, sa, ca, d], [0, 0, 0, 1.]])
    return t


def step(env, state, action):
    return env.step(np.asarray(action, dtype=np.float32))[0]


def command(env, state, base, qobs, grip, repeats=30):
    for _ in range(repeats):
        cur = np.array([val(state, "robot", "pos_base_x"),
                        val(state, "robot", "pos_base_y"),
                        val(state, "robot", "pos_base_rot")])
        cq = np.array([val(state, "robot", "joint_%d" % i)
                       for i in range(1, 8)])
        action = np.zeros(11)
        action[:3] = np.clip(np.asarray(base) - cur, -.2, .2)
        action[3:10] = np.clip(qobs - cq, -.2, .2)
        action[10] = grip
        state = step(env, state, action)
        if max(np.max(abs(np.asarray(base)-cur)), np.max(abs(qobs-cq))) < 2e-3:
            break
    return state


def solutions():
    qseed = np.array([0., -.35, 0., -2.5, 0., 2.2716, .7854])
    # Try multiple radial reaches/heights and both vertical tool directions.
    for z in np.linspace(-.35, .35, 15):
        for radius in (.25, .4, .55, .7):
            for down in (-1., 1.):
                target = np.array([radius, 0., z])
                def residual(q):
                    t = fk(q)
                    return np.r_[5*(t[:3, 3]-target),
                                  1.5*(t[:3, 2]-[0, 0, down])]
                sol = least_squares(residual, qseed, bounds=(-3.1, 3.1),
                                    max_nfev=250).x
                err = np.linalg.norm(residual(sol))
                if err < .18:
                    yield z, radius, down, sol, fk(sol)[:3, 3]


def run(seed=0, target="box0", start=0, stop=120):
    candidates = list(solutions())[start:stop]
    # Each candidate uses a fresh episode so its movements never disturb target.
    for number, (z, radius, down, qphys, ee) in enumerate(candidates, start):
        # DH x/y frame may differ by a signed axis permutation from world frame.
        for mapping in range(4):
            env = make_env(); state, _ = env.reset(seed=seed)
            tx, ty = val(state, target, "pose_x"), val(state, target, "pose_y")
            ex, ey = ee[:2]
            offsets = [(ex, ey), (-ex, -ey), (-ey, ex), (ey, -ex)]
            ox, oy = offsets[mapping]
            base = [tx-ox, ty-oy, 0.]
            qobs = qphys - JOINT_OFFSET
            state = command(env, state, base, qobs, 1.)
            # Close for two stable frames.
            state = command(env, state, base, qobs, -1., repeats=2)
            active = val(state, "robot", "grasp_active") > .5
            if active:
                before = [val(state, target, "pose_"+c) for c in "xyz"]
                move = np.zeros(11); move[0] = .2; move[1] = .1; move[10] = -1
                state = step(env, state, move)
                after = [val(state, target, "pose_"+c) for c in "xyz"]
                print(json.dumps({"FOUND": True, "candidate": number,
                      "z": z, "radius": radius, "down": down,
                      "mapping": mapping, "q_phys": qphys.tolist(),
                      "q_obs": qobs.tolist(), "base": base,
                      "object_before": before, "object_after_move": after,
                      "grasp_tf": [val(state, "robot", "grasp_tf_"+c)
                                   for c in "xyz"]}))
                env.close(); return True
            env.close()
        if number % 10 == 0:
            print("searched", number, "z/r/down", z, radius, down, flush=True)
    print("NONE", start, stop, "of", len(list(solutions())))
    return False


def full_range(seed=0, target="box0", search_seed=0):
    """Search actual discovered joint ranges, not a symmetric fake range."""
    rng = np.random.default_rng(search_seed)
    lo = np.array([0., -.35, -math.pi, -2.5, 0., -.87, math.pi/2])
    hi = lo + np.array([5.2, 2.41, 2.058, 2.66, 5.2, 2.23, 6.77])
    env = make_env(); state, _ = env.reset(seed=seed)
    tx, ty = val(state, target, "pose_x"), val(state, target, "pose_y")
    for attempt in range(42):
        # Broad independent arm posture and base offset.  The latter translates
        # the entire gripper workspace without altering its height/orientation.
        qobs = rng.uniform(lo, hi)
        radius = rng.uniform(.05, .95)
        angle = rng.uniform(-math.pi, math.pi)
        base = [tx + radius*math.cos(angle), ty + radius*math.sin(angle),
                rng.uniform(-math.pi, math.pi)]
        state = command(env, state, base, qobs, 1., repeats=35)
        state = command(env, state, base, qobs, -1., repeats=2)
        if val(state, "robot", "grasp_active") > .5:
            realized_q = [val(state, "robot", "joint_%d" % i) for i in range(1, 8)]
            realized_base = [val(state, "robot", "pos_base_x"),
                             val(state, "robot", "pos_base_y"),
                             val(state, "robot", "pos_base_rot")]
            before = [val(state, target, "pose_"+c) for c in "xyz"]
            move = np.zeros(11); move[:2] = [.2, .1]; move[10] = -1
            state = step(env, state, move)
            after = [val(state, target, "pose_"+c) for c in "xyz"]
            print(json.dumps({"FOUND": True, "search_seed": search_seed,
                  "attempt": attempt, "q_obs": realized_q,
                  "base": realized_base,
                  "base_minus_object": [realized_base[0]-tx, realized_base[1]-ty],
                  "object_before": before, "object_after_move": after,
                  "grasp_tf": [val(state, "robot", "grasp_tf_"+c)
                               for c in "xyz"]}))
            env.close(); return True
        print("attempt", attempt, flush=True)
    env.close(); print("FULL_NONE", search_seed); return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="box0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=120)
    parser.add_argument("--full-range", action="store_true")
    parser.add_argument("--search-seed", type=int, default=0)
    args = parser.parse_args()
    if args.full_range:
        full_range(args.seed, args.target, args.search_seed)
    else:
        run(args.seed, args.target, args.start, args.stop)
