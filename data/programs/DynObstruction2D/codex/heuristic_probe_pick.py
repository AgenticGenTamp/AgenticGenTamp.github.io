import math
import numpy as np
from env_client import make_env


def clip(x, m): return max(-m, min(m, x))


def run(seed, reach):
    env = make_env(); s, _ = env.reset(seed=seed); typ = env.observation_space.get_type
    r = s.get_objects(typ("kin_robot"))[0]; b = s.get_objects(typ("target_block"))[0]
    print("features", {f:round(s.get(r,f),3) for f in ("base_radius","arm_joint","arm_length","gripper_base_width","gripper_base_height","finger_gap","finger_height","finger_width")})
    phase = 0
    for t in range(160):
        rx, ry, th = [s.get(r,f) for f in ("x","y","theta")]
        bx, by = [s.get(b,f) for f in ("x","y")]
        # face right, place nominal grasp center at block center, close after settled
        act = np.zeros(5, dtype=np.float32)
        if phase == 0:
            act[1] = clip(by-ry, .049); act[2] = clip(-th, .196)
            if abs(by-ry) < .025 and abs(th) < .03: phase = 1
        elif phase == 1:
            act[0] = clip(bx-reach-rx, .049)
            if bx-rx <= reach+.015: phase = 2
        elif phase == 2:
            act[4] = -.02
            if s.get(r,"finger_gap") < .13: phase = 3
        else:
            act[1] = .04
        act = np.clip(act, env.action_space.low, env.action_space.high).astype(env.action_space.dtype)
        s, rew, term, trunc, _ = env.step(act)
        if t % 10 == 0 or s.get(b,"held") > .5:
            print(seed, reach, t, "r", [round(s.get(r,f),2) for f in ("x","y","theta","arm_joint","finger_gap")], "b", [round(s.get(b,f),2) for f in ("x","y","held")])
        if term or trunc: break
    env.close()


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]), float(sys.argv[2]))
