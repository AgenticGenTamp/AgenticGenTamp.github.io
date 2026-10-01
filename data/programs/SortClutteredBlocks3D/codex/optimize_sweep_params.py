"""Evaluate small perturbations of the policy's first calibrated red sweep.

This is deliberately a test-only wrapper: it mutates the cached contact path
and clips shoulder motion without changing the submitted approach.
"""
import argparse
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def xy(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, "x"), state.get(obj, "y")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lateral", type=float, default=0.042)
    parser.add_argument("--reach", type=float, default=-0.75)
    parser.add_argument("--q1", type=float, default=0.80)
    parser.add_argument("--steps", type=int, default=275)
    parser.add_argument("--red-second", action="store_true")
    parser.add_argument("--target-only-contact", action="store_true")
    parser.add_argument("--force-x-first", action="store_true")
    args = parser.parse_args()
    env = make_env()
    state, info = env.reset(seed=args.seed, options={"object_count": 4})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    red_names = []
    for name in state.get_object_names():
        if not name.startswith("cube") or not name[4:].isdigit():
            continue
        if (int(name[4:]) - policy.index_offset) % 4 == 0:
            red_names.append(name)
    initial = {name: xy(state, name) for name in red_names}
    target = policy.bin_targets["red"].copy()
    if args.force_x_first:
        policy.red_x_first.update(red_names)
        args.red_second = True
    if args.red_second and policy.plan:
        first_red = policy.plan[0][0]
        if len(policy.plan) < 2 or policy.plan[1] != (first_red, "second"):
            policy.plan.insert(1, (first_red, "second"))
    context_seen = None
    reward = 0.0
    terminated = truncated = False
    for step in range(args.steps):
        action = policy.get_action(state)
        if (args.target_only_contact and policy.hit_base is not None and
                policy.contact_xy is not None and policy.push_context is not None):
            active_name = policy.push_context[0]
            if np.linalg.norm(xy(state, active_name) -
                              policy.contact_xy[active_name]) <= 0.005:
                # Ignore an incidental neighbor hit and keep advancing until
                # the intended cube itself has measurably moved.
                policy.hit_base = None
        if policy.push_context is not None and context_seen is not policy.push_context:
            context_seen = policy.push_context
            name, outer, contact, start, turn, old, target_xy = context_seen
            yaw = contact[2]
            c, s = np.cos(yaw), np.sin(yaw)
            cube_at_plan = old[name]
            def pose(local_x):
                return np.array([cube_at_plan[0] + c * local_x - s * args.lateral,
                                 cube_at_plan[1] + s * local_x + c * args.lateral,
                                 yaw])
            # Preserve the safe outer waypoint, changing only its lateral line.
            outer[:] = pose(-0.97)
            contact[:] = pose(args.reach)
        # During the q1 sweep, clip the commanded velocity so the realized
        # shoulder angle cannot run beyond the candidate endpoint.
        robot = state.get_object_from_name("robot")
        q1 = state.get(robot, "pos_arm_joint1")
        if q1 >= args.q1 and action[3] > 0:
            action[3] = min(0.0, 0.8 * (args.q1 - q1))
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    result = {name: round(float(np.linalg.norm(xy(state, name) - target)), 4)
              for name in red_names}
    moved = {name: np.round(xy(state, name) - initial[name], 4).tolist()
             for name in red_names}
    robot = state.get_object_from_name("robot")
    final_q1 = state.get(robot, "pos_arm_joint1")
    print("RESULT", args.seed, args.lateral, args.reach, args.q1,
          step + 1, round(float(reward), 4), terminated, result, moved,
          "q1", round(float(final_q1), 4),
          "xfirst", sorted(policy.red_x_first))
    env.close()


if __name__ == "__main__":
    main()
