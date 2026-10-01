"""Search wrist-roll postures that permit lowering at the north tangent pose."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import itertools


def setup(seed):
    env = make_env(); state, info = env.reset(seed=seed)
    p = GeneratedApproach(env.action_space, env.observation_space, {}); p.reset(state, info)
    while p.stage < 5: state, *_ = env.step(p.get_action(state))
    p.get_action(state); target = p.target(p.green)
    for _ in range(2): state, *_ = env.step(p.motion(state, p.robot(state), 1, lift=True))
    theta = p.theta; p.theta = p.g(state, "robot", "base_rot")
    for _ in range(3): state, *_ = env.step(p.motion(state, target, 1, lift=True))
    p.theta = theta
    return env, state, p


def change(env, state, index, delta):
    left = delta
    while abs(left) > .001:
        action = np.zeros(11, np.float32); action[10] = 1
        action[index] = np.clip(left, -.2, .2)
        old = state; state, *_ = env.step(action); left -= action[index]
    return state


for j5, j7 in itertools.product((-1.2, -.8, -.4, 0., .4, .8, 1.2), repeat=2):
    env, state, p = setup(101)
    state = change(env, state, 7, j5); state = change(env, state, 9, j7)
    reached = p.g(state, "robot", "joint_2")
    for _ in range(10):
        action = np.zeros(11, np.float32); action[4] = .025; action[10] = 1
        state, *_ = env.step(action); reached = p.g(state, "robot", "joint_2")
    action = np.zeros(11, np.float32); action[10] = -1
    state, *_ = env.step(action); held = p.g(state, "robot", "grasp_active")
    if reached > .09 or held:
        print("rolls", j5, j7, "q2", round(reached, 4), "held", held)
    env.close()
