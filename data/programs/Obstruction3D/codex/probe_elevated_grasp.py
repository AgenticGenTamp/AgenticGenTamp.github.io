from env_client import make_env
from approach import GeneratedApproach
import numpy as np


def f(s, name, feat):
    return float(s.get(s.get_object_from_name(name), feat))


def setup(seed=21, max_steps=1000, min_z=.1077, verbose=True):
    env = make_env(); s, info = env.reset(seed=seed)
    p = GeneratedApproach(env.action_space, env.observation_space, {})
    p.reset(s, info)
    last = None
    for t in range(max_steps):
        s, r, done, trunc, info = env.step(p.get_action(s))
        held = f(s, "robot", "grasp_active") > .5
        key = (p.stage, p.i, held)
        if verbose and (key != last or (not held and f(s, "target_block", "pose_z") > .105)):
            q = [round(f(s, "robot", "joint_%d" % i), 4) for i in range(1, 8)]
            xyz = [round(f(s, "target_block", "pose_" + a), 5) for a in "xyz"]
            print("TRACE", t + 1, key, "xyz", xyz, "base", [round(f(s,"robot",x),4) for x in ("pos_base_x","pos_base_y")], "q", q)
        last = key
        if (not held and f(s, "target_block", "pose_z") > min_z and p.stage in ("plan", "reposition", "approach", "search")):
            return env, s, p, t + 1
        if done or trunc: break
    return env, s, p, t + 1


def trial(q2, q4, offx=0., offy=0.):
    env, s, p, t = setup(verbose=False)
    p._make_plan(s, "target_block")
    bx, by = p.base[0] + offx, p.base[1] + offy
    print("START", t, "obj", [round(f(s,"target_block","pose_"+a),6) for a in "xyz"],
          "basegoal", [round(bx,6),round(by,6)], "q1", round(p.q_start,6), "posture", q2,q4)
    for k in range(20):
        a = p._act(s, bx, by, p.q_start, q2, q4, p.Q6, p.Q7, grip=-1.)
        s, r, done, trunc, info = env.step(a)
        held = f(s,"robot","grasp_active") > .5
        print("TRY", k+1, "held", held, "base", [round(f(s,"robot",x),4) for x in ("pos_base_x","pos_base_y")],
              "q", [round(f(s,"robot","joint_%d"%i),4) for i in (1,2,4,6,7)])
        if held: break
    env.close()
    return held


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        trial(*map(float, sys.argv[1:]))
    else:
        env, s, p, t = setup()
        print("READY", t, p.stage, "z", f(s,"target_block","pose_z"))
        env.close()
