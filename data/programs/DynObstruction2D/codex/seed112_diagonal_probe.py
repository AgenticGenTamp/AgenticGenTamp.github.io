"""Targeted mirrored diagonal rake experiments for left-boundary seed 112."""
import math
import sys

import numpy as np

from env_client import make_env


def run(angle=-1.94, y_offset=.215, speed=.010, close=False, base_x=None):
    env = make_env()
    state, _ = env.reset(seed=112)
    typ = env.observation_space.get_type
    robot = state.get_objects(typ("kin_robot"))[0]
    block = state.get_objects(typ("target_block"))[0]
    surface = state.get_objects(typ("target_surface"))[0]
    get = lambda obj, feature: float(state.get(obj, feature))
    terminal = truncated = False

    def step(action):
        nonlocal state, terminal, truncated
        action = np.clip(np.asarray(action, dtype=float),
                         env.action_space.low + 1e-7,
                         env.action_space.high - 1e-7)
        state, _, terminal, truncated, _ = env.step(action)

    # Retract and open at a safe height.
    for _ in range(30):
        step([0, np.clip(1.10-get(robot,"y"),-.049,.049), 0, -.099, .019])
    # Mirror of the successful right-boundary diagonal geometry.  The base is
    # just right of the cargo center while the arm points down-left.
    target_x = (get(block,"x") - get(block,"width")/2 + .24
                if base_x is None else base_x)
    for _ in range(72):
        error = (angle-get(robot,"theta")+math.pi)%(2*math.pi)-math.pi
        step([np.clip(target_x-get(robot,"x"),-.049,.049), 0,
              np.clip(error,-.19,.19), -.099, .019])
        if abs(target_x-get(robot,"x")) < .012 and abs(error) < .025:
            break
    for _ in range(35):
        target_y = get(block,"y") + y_offset
        step([0, np.clip(target_y-get(robot,"y"),-.049,.049), 0, -.099, 0])
        if abs(target_y-get(robot,"y")) < .012:
            break
    if close:
        for _ in range(10):
            step([0,0,0,0,-.019])
    start_x = get(block,"x")
    min_y = get(block,"y")
    for push_step in range(140):
        step([speed,0,0,0,0])
        min_y = min(min_y, get(block,"y"))
        if terminal or truncated:
            break
    result = (terminal, push_step+1, get(block,"x"), get(block,"y"),
              get(block,"theta"), get(block,"x")-start_x, min_y,
              get(robot,"x"), get(robot,"y"))
    print("angle", angle, "yoff", y_offset, "speed", speed, "close", close,
          "base_x", base_x, "=>",
          tuple(round(v,3) if isinstance(v,float) else v for v in result),
          "surface", round(get(surface,"x"),3), flush=True)
    env.close()
    return result


if __name__ == "__main__":
    angle = float(sys.argv[1]) if len(sys.argv)>1 else -1.94
    y_offset = float(sys.argv[2]) if len(sys.argv)>2 else .215
    speed = float(sys.argv[3]) if len(sys.argv)>3 else .010
    close = bool(int(sys.argv[4])) if len(sys.argv)>4 else False
    base_x = float(sys.argv[5]) if len(sys.argv)>5 else None
    run(angle, y_offset, speed, close, base_x)
