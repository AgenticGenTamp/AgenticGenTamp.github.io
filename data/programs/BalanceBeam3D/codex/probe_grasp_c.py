"""Targeted seed-0 grasp grid for the low q2/q4 posture."""
import argparse
import numpy as np

from env_client import make_env


OBJ = ((0, "large"), (54, "small1"), (70, "small2"))


def positions(s):
    return {name: np.array(s[i:i + 3], dtype=float) for i, name in OBJ}


def move_to(env, s, base_target=None, joint_targets=None, grip=0.0, limit=100):
    """Bang-bang velocity control to observed base/joint targets."""
    for n in range(limit):
        a = np.zeros(11, np.float32)
        a[10] = grip
        errs = []
        if base_target is not None:
            for k in range(2):
                e = float(base_target[k] - s[16 + k]); errs.append(abs(e))
                a[k] = np.clip(0.8 * e, -0.1, 0.1)
        if joint_targets:
            for joint, target in joint_targets.items():
                e = float(target - s[19 + joint]); errs.append(abs(e))
                a[3 + joint] = np.clip(0.8 * e, -0.1, 0.1)
        if errs and max(errs) < 0.012:
            return s, n
        s, r, term, trunc, info = env.step(a)
        if term or trunc:
            return s, n + 1
    return s, limit


def trial(dx, dy, close_value, q2_target=1.58):
    env = make_env()
    s, _ = env.reset(seed=0)
    initial = positions(s)
    base0 = s[16:18].astype(float).copy()
    open_value = 1.0 - close_value

    # Translate while the arm remains safely raised, then descend to fixed posture.
    s, nb = move_to(env, s, base0 + np.array([dx, dy]), grip=open_value, limit=25)
    s, na = move_to(env, s, joint_targets={1: q2_target, 3: -1.46},
                    grip=open_value, limit=110)
    at_bottom = positions(s)
    bottom_robot = np.round(s[16:27], 4).tolist()

    a = np.zeros(11, np.float32); a[10] = close_value
    for _ in range(8):
        s, r, term, trunc, info = env.step(a)
    after_close = positions(s)

    # Retract shoulder while maintaining the closing command.
    s, nl = move_to(env, s, joint_targets={1: 0.95, 3: -1.46},
                    grip=close_value, limit=35)
    lifted = positions(s)
    env.close()

    result = {}
    success = False
    for name in initial:
        d_bottom = at_bottom[name] - initial[name]
        d_lift = lifted[name] - after_close[name]
        result[name] = (np.round(d_bottom, 3).tolist(),
                        np.round(d_lift, 3).tolist(),
                        np.round(lifted[name], 3).tolist())
        # Require upward motion during the lift, not merely a knock during descent.
        if d_lift[2] > 0.025 and lifted[name][2] > 0.035:
            success = True
    return success, (nb, na, nl), result, bottom_robot, np.round(s[16:27], 3).tolist()


def vertical_trial(x_gap, y_gap):
    env = make_env(); s, _ = env.reset(seed=0)
    initial = positions(s)
    large = initial["large"]
    base_target = np.array([large[0] - x_gap, large[1] - y_gap])
    s, nb = move_to(env, s, base_target=base_target, grip=0.0, limit=25)
    s, na = move_to(env, s, joint_targets={1: 1.65, 3: -1.46, 5: -.4},
                    grip=0.0, limit=105)
    bottom = positions(s); bottom_robot = np.round(s[16:27], 4).tolist()
    a = np.zeros(11, np.float32); a[10] = 1.0
    for _ in range(8): s, r, term, trunc, info = env.step(a)
    closed = positions(s); close_robot = np.round(s[16:27], 4).tolist()
    s, nl = move_to(env, s, joint_targets={1: 1.0, 3: -1.46, 5: -.4},
                    grip=1.0, limit=38)
    end = positions(s); final_robot = np.round(s[16:27], 4).tolist()
    env.close()
    result = {}
    moved = False; picked = False
    for name in initial:
        db = bottom[name] - initial[name]
        dc = closed[name] - bottom[name]
        dl = end[name] - closed[name]
        result[name] = (np.round(db, 4).tolist(), np.round(dc, 4).tolist(),
                        np.round(dl, 4).tolist(), np.round(end[name], 4).tolist())
        moved |= np.linalg.norm(db) > .002 or np.linalg.norm(dc) > .002 or np.linalg.norm(dl) > .002
        picked |= dl[2] > .025 and end[name][2] > .035
    return moved, picked, base_target, (nb, na, nl), result, bottom_robot, close_robot, final_robot


def fine_trial(x_gap, y_gap):
    env = make_env(); s, _ = env.reset(seed=0)
    initial = positions(s); large0 = initial["large"].copy()
    target = np.array([large0[0] - x_gap, large0[1] - y_gap])
    s, nb = move_to(env, s, base_target=target, grip=0.0, limit=25)
    jt = {1: 1.65, 3: -1.46, 5: -.4}
    s, na = move_to(env, s, joint_targets=jt, grip=0.0, limit=105)
    bottom_robot = np.round(s[16:27], 5).tolist()
    before_close = positions(s)
    a = np.zeros(11, np.float32); a[10] = 1.0
    for _ in range(8): s, r, term, trunc, info = env.step(a)
    after_close = positions(s); close_robot = np.round(s[16:27], 5).tolist()
    nudge = after_close["large"] - before_close["large"]
    recentered = False
    if np.linalg.norm(nudge[:2]) > .002 and after_close["large"][2] < .035:
        # Follow the displaced cube in world x/y, reopen, and close at the same pose.
        a[10] = 0.0
        for _ in range(5): s, r, term, trunc, info = env.step(a)
        new_target = s[16:18].astype(float) + nudge[:2]
        s, _ = move_to(env, s, base_target=new_target, joint_targets=jt,
                       grip=0.0, limit=18)
        a[:] = 0; a[10] = 1.0
        for _ in range(8): s, r, term, trunc, info = env.step(a)
        after_close = positions(s); close_robot = np.round(s[16:27], 5).tolist()
        recentered = True

    zref = after_close["large"][2]
    streak = 0; trace = []; picked = False
    for k in range(100):
        a = np.zeros(11, np.float32); a[4] = -.02; a[10] = 1.0
        s, r, term, trunc, info = env.step(a)
        p = np.array(s[0:3], float); dz = p[2] - zref
        if k < 10 or k % 10 == 9 or dz > .01:
            trace.append((k + 1, np.round(p, 4).tolist(), round(float(s[20]), 4)))
        streak = streak + 1 if dz > .02 else 0
        if streak >= 3:
            picked = True; break
    end = positions(s); final_robot = np.round(s[16:27], 5).tolist()
    env.close()
    delta = {name: (np.round(after_close[name] - before_close[name], 4).tolist(),
                    np.round(end[name] - after_close[name], 4).tolist(),
                    np.round(end[name], 4).tolist()) for name in initial}
    return picked, recentered, target, (nb, na, k + 1), delta, bottom_robot, close_robot, final_robot, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--close", type=float, default=1.0)
    p.add_argument("--one", action="store_true")
    p.add_argument("--corrected", action="store_true")
    p.add_argument("--first", action="store_true")
    p.add_argument("--vertical", action="store_true")
    p.add_argument("--fine", action="store_true")
    args = p.parse_args()
    if args.fine:
        for xgap in (.44, .42, .40):
            for ygap in (.03, .01, .05):
                picked, recent, bt, counts, delta, br, cr, fr, trace = fine_trial(xgap, ygap)
                print("PICKUP" if picked else "miss", "gaps",(xgap,ygap),
                      "base_target",np.round(bt,5).tolist(),"recentered",recent,
                      "steps",counts,"objects(close,lift,end)",delta,
                      "bottom_robot",br,"close_robot",cr,"final_robot",fr,
                      "lift_trace",trace,flush=True)
                if picked:
                    return
        return
    if args.vertical:
        for xgap in (.46, .44, .48):
            for ygap in (.01, 0., .02):
                moved, picked, bt, counts, delta, br, cr, fr = vertical_trial(xgap, ygap)
                print("PICKUP" if picked else ("CONTACT" if moved else "miss"),
                      "gaps",(xgap,ygap),"base_target",np.round(bt,4).tolist(),
                      "steps",counts,"objects(desc,close,lift,end)",delta,
                      "bottom_robot",br,"close_robot",cr,"final_robot",fr,flush=True)
                if moved:
                    return
        return
    # Center-first grid; dy is base displacement and spans the requested +/-8 cm.
    candidates = [(x, y) for x in (.39, .34, .44) for y in (0., -.04, .04, -.08, .08)]
    if args.one:
        candidates = [(.34, .20, 1.58)]
    elif args.corrected:
        candidates = [(.38, y, q2) for q2 in (1.8, 1.7, 1.9) for y in (-.05, 0.)]
    else:
        candidates = [(x, y, 1.58) for x, y in candidates]
    if args.first:
        candidates = candidates[:1]
    for dx, dy, q2 in candidates:
        ok, counts, delta, bottom_robot, robot = trial(dx, dy, args.close, q2)
        print("SUCCESS" if ok else "miss", "dxdy", (dx, dy), "close", args.close,
              "q2target", q2, "steps", counts,
              "objects(bottom,lift,end)", delta, "bottom_robot", bottom_robot,
              "final_robot", robot,
              flush=True)
        contact = any(any(abs(v) > .002 for v in values[0]) for values in delta.values())
        if ok or (args.corrected and contact):
            break


if __name__ == "__main__":
    main()
