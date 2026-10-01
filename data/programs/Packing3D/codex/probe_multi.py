"""Diagnose multi-part policy ordering and rack interference."""
import argparse

from approach import GeneratedApproach
from env_client import make_env


def val(state, name, feature):
    return state.get(state.get_object_from_name(name), feature)


def describe(state, name):
    obj = state.get_object_from_name(name)
    kind = "cuboid"
    if obj.type.name == "Kinematic3DTriangle":
        kind = "tri%d" % round(val(state, name, "triangle_type"))
    return (kind, round(val(state, name, "pose_x"), 4),
            round(val(state, name, "pose_y"), 4),
            round(val(state, name, "pose_z"), 4),
            round(val(state, name, "grasp_active")))


def run(seed, count, limit):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": count})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    parts = sorted(n for n in state.get_object_names() if n.startswith("part"))
    initial = {p: describe(state, p) for p in parts}
    events = []
    old = (policy.phase, policy.target)
    terminated = truncated = False
    for step in range(limit):
        action = policy.get_action(state)
        now = (policy.phase, policy.target)
        if now != old:
            events.append((step, now, getattr(policy, "slot_key", None),
                           describe(state, now[1]) if now[1] else None))
            old = now
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    final = {p: describe(state, p) for p in parts}
    print("seed=%d count=%d steps=%d term=%s init=%r final=%r" %
          (seed, count, step + 1, terminated, initial, final))
    print(" events=%r placed=%r slots=%r failures=%r deferred=%r" %
          (events, policy.placed, policy.used_slots, policy.failures,
           sorted(policy.deferred)))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("seeds", nargs="*", type=int, default=list(range(5)))
    args = parser.parse_args()
    for episode_seed in args.seeds:
        run(episode_seed, args.count, args.limit)
