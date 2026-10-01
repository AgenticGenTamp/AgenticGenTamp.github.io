"""Try collision-safe staged returns after moving the southeast blocker."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np


def drive(policy, env, state, target, lift=True, q4=.08, limit=30):
    for _ in range(limit):
        if policy.at(state, target):
            return state, True
        state, _, done, truncated, _ = env.step(
            policy.motion(state, target, 1, lift=lift, q4add=q4))
        if done or truncated:
            return state, False
    return state, False


for seed in [74, 79, 120, 176, 196]:
    env = make_env(); state, info = env.reset(seed=seed)
    p = GeneratedApproach(env.action_space, env.observation_space, {}); p.reset(state, info)
    # Run only through the blocker release.
    for step in range(80):
        state, _, done, truncated, _ = env.step(p.get_action(state))
        if p.stage == 5:
            break
    print("seed", seed, "drop", p.robot(state), "out", p.out)

    # Extend q4 in the collision-free blocker drop pose, keeping the arm lifted.
    for _ in range(4):
        action = np.zeros(11, np.float32)
        action[6] = np.clip(p.Q[3] + .08 - p.g(state, "robot", "joint_4"), -.2, .2)
        action[10] = 1
        state, _, _, _, _ = env.step(action)
    print(" q4", p.g(state, "robot", "joint_4"))

    # First translate horizontally south of all pen geometry, then head north.
    final = p.target(p.green) + np.array([.10, -.10])
    corner = np.array([final[0], p.robot(state)[1]])
    state, ok1 = drive(p, env, state, corner)
    state, ok2 = drive(p, env, state, final)
    print(" return", ok1, ok2, p.robot(state), "final", final)

    # Lower at the final base pose, close, and see if the green is acquired.
    for _ in range(8):
        state, _, _, _, _ = env.step(p.motion(state, final, 1, q4add=.08))
    action = np.zeros(11, np.float32); action[10] = -1
    state, _, _, _, _ = env.step(action)
    tool = [p.g(state, "robot", "grasp_tf_"+x) for x in "xyz"]
    print(" held", p.g(state, "robot", "grasp_active"), "tool", np.round(tool, 4),
          "tool-green", np.round(np.array(tool[:2])-p.green, 4),
          "green", [p.g(state, "green0", "pose_"+x) for x in "xyz"])
    env.close()
