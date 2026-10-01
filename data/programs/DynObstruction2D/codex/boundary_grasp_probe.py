"""Probe a centered top-down grasp on narrow right-boundary cargo."""
import math
import numpy as np
from env_client import make_env


def clip_action(e, values):
    return np.clip(np.asarray(values, dtype=float), e.action_space.low + 1e-7,
                   e.action_space.high - 1e-7)


def run(seed):
    e = make_env(); s, _ = e.reset(seed=seed); T = e.observation_space.get_type
    r = s.get_objects(T("kin_robot"))[0]
    b = s.get_objects(T("target_block"))[0]
    g = lambda o, f: float(s.get(o, f))
    bx0, by0 = g(b, "x"), g(b, "y")

    # Retract and open at safe height.
    for _ in range(25):
        s, *_ = e.step(clip_action(e, [0, np.clip(1.12-g(r,"y"),-.049,.049), 0, -.099, .019]))
    # Center the vertical arm directly over the cargo.
    for _ in range(70):
        err = (-math.pi/2-g(r,"theta")+math.pi)%(2*math.pi)-math.pi
        s, *_ = e.step(clip_action(e, [np.clip(g(b,"x")-g(r,"x"),-.049,.049), 0,
                                           np.clip(err,-.19,.19), -.099, .019]))
        if abs(g(b,"x")-g(r,"x")) < .01 and abs(err) < .02:
            break
    # With the arm retracted, base-to-jaw-center offset is approximately .40.
    jaw_base_y = g(b,"y") + .40
    for _ in range(35):
        s, *_ = e.step(clip_action(e, [np.clip(g(b,"x")-g(r,"x"),-.049,.049),
                                           np.clip(jaw_base_y-g(r,"y"),-.049,.049), 0, -.099, .019]))
        if abs(jaw_base_y-g(r,"y")) < .01:
            break
    before = (g(b,"x"), g(b,"y"), g(b,"held"), g(r,"finger_gap"))
    for _ in range(14):
        s, *_ = e.step(clip_action(e, [0, 0, 0, 0, -.019]))
    closed = (g(b,"x"), g(b,"y"), g(b,"held"), g(r,"finger_gap"))
    for _ in range(30):
        s, *_ = e.step(clip_action(e, [-.025, 0, 0, 0, 0]))
    moved = (g(b,"x"), g(b,"y"), g(b,"held"), g(r,"finger_gap"))
    print(seed, "width", round(g(b,"width"),3),
          "before", tuple(round(v,3) for v in before),
          "closed", tuple(round(v,3) for v in closed),
          "moved", tuple(round(v,3) for v in moved),
          "dx", round(moved[0]-bx0,3), "dy", round(moved[1]-by0,3), flush=True)
    e.close()


for seed in (50, 54):
    run(seed)
