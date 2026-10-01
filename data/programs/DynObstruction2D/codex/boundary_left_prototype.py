"""Prototype a top-down pick/place for tight leftward placements.

This exploration script is intentionally separate from approach.py.
"""
import math
import sys

import numpy as np

from env_client import make_env


def run(seed, verbose=False):
    env = make_env()
    state, info = env.reset(seed=seed)
    typ = env.observation_space.get_type
    robot = state.get_objects(typ("kin_robot"))[0]
    block = state.get_objects(typ("target_block"))[0]
    surface = state.get_objects(typ("target_surface"))[0]
    get = lambda obj, feature: float(state.get(obj, feature))

    def step(values, n=1, label="", done_when=None):
        nonlocal state
        terminal = False
        for _ in range(n):
            action = np.clip(np.asarray(values(), dtype=float),
                             env.action_space.low + 1e-7,
                             env.action_space.high - 1e-7)
            state, _, terminal, truncated, _ = env.step(action)
            if terminal or truncated:
                break
            if done_when is not None and done_when():
                break
        if verbose:
            print(label, "robot", round(get(robot,"x"),3), round(get(robot,"y"),3),
                  "block", round(get(block,"x"),3), round(get(block,"y"),3),
                  "theta", round(get(block,"theta"),3), "held", get(block,"held"),
                  "gap", round(get(robot,"finger_gap"),3))
        return terminal

    # Safely retract/open while going above the object.
    step(lambda: [0, np.clip(1.12-get(robot,"y"),-.049,.049), 0, -.099, .019], 30, "raise")
    step(lambda: [np.clip(get(block,"x")-get(robot,"x"),-.049,.049), 0,
                  np.clip(((-math.pi/2-get(robot,"theta")+math.pi)%(2*math.pi)-math.pi),-.19,.19),
                  -.099, .019], 70, "align",
         lambda: abs(get(block,"x")-get(robot,"x")) < .01 and
         abs((-math.pi/2-get(robot,"theta")+math.pi)%(2*math.pi)-math.pi) < .02)
    # Base y=block y+0.40 puts the open fingers around the block.
    step(lambda: [0, np.clip(get(block,"y")+.40-get(robot,"y"),-.049,.049),0,-.099,.019],
         35, "lower", lambda: abs(get(block,"y")+.40-get(robot,"y")) < .01)
    step(lambda: [0,0,0,0,-.019], 14, "close")
    # Lift well clear of both floor and pad lip, translate cargo over pad.
    step(lambda: [0, np.clip(.62-get(block,"y"),-.035,.035),0,0,0], 30, "lift")
    step(lambda: [np.clip(get(surface,"x")-get(block,"x"),-.025,.025),0,0,0,0], 45, "translate")
    target_y = get(surface,"y") + get(surface,"height")/2 + get(block,"height")/2 + .006
    step(lambda: [0,np.clip(target_y-get(block,"y"),-.025,.025),0,0,0], 30, "lower cargo")
    terminal = step(lambda: [0,0,0,0,.019], 14, "release")
    if not terminal:
        # Withdraw upward without disturbing the settling cargo.
        terminal = step(lambda: [0,.035,0,0,0], 20, "withdraw")
    print(seed, bool(terminal), "block", round(get(block,"x"),3), round(get(block,"y"),3),
          "theta", round(get(block,"theta"),3), "surface", round(get(surface,"x"),3))
    env.close()
    return terminal


if __name__ == "__main__":
    for argument in sys.argv[1:] or ["35", "1", "8"]:
        run(int(argument), verbose=True)
