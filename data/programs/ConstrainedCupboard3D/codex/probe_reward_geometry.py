"""Probe cupboard/object geometry, passive reward, and rigid-body contacts."""
import argparse
import math
import numpy as np
from env_client import make_env


def f(state, obj, name):
    return float(state.get(obj, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--mode", choices=("geometry", "base_push", "cupboard"), default="geometry")
    args = parser.parse_args()
    env = make_env()
    try:
        state, info = env.reset(seed=args.seed, options={"object_count": args.count})
        fix_type = env.observation_space.get_type("mujoco_fixture")
        mov_type = env.observation_space.get_type("mujoco_movable_object")
        fixtures = list(state.get_objects(fix_type))
        rods = list(state.get_objects(mov_type))
        print("meta", info, "fixtures", len(fixtures), "rods", len(rods))
        print("fixture_xyz", sorted((round(f(state,o,"x"),4), round(f(state,o,"y"),4),
                                     round(f(state,o,"z"),4)) for o in fixtures))
        for o in sorted(rods, key=lambda x: x.name):
            yaw = 2*math.atan2(f(state,o,"qz"), f(state,o,"qw"))
            print(o.name, "xyz", *(round(f(state,o,k),4) for k in ("x","y","z")),
                  "yaw", round(yaw,4), "bb", *(round(f(state,o,k),4) for k in ("bb_x","bb_y","bb_z")))

        robot = state.get_object_from_name("robot")
        original = {o.name: np.array([f(state,o,k) for k in ("x","y","z")]) for o in rods}
        if args.mode == "geometry":
            rewards = []
            for _ in range(5):
                state, reward, term, trunc, _ = env.step(np.zeros(11, np.float32))
                rewards.append(reward)
            print("passive", rewards, "done", term, trunc)
        else:
            target = rods[0] if args.mode == "base_push" else fixtures[0]
            tx, ty = f(state,target,"x"), f(state,target,"y")
            # Drive through target along x after aligning in y.  The arm remains home.
            for step in range(45):
                bx, by = f(state,robot,"pos_base_x"), f(state,robot,"pos_base_y")
                action = np.zeros(11, np.float32)
                if step < 10:
                    action[1] = np.clip((ty-by)/.87, -.1, .1)
                else:
                    action[0] = .1
                state, reward, term, trunc, _ = env.step(action)
                moved = {o.name: round(float(np.linalg.norm(np.array([f(state,o,k) for k in ("x","y","z")])-original[o.name])),5) for o in rods}
                if step % 5 == 0:
                    print(step, "base", round(f(state,robot,"pos_base_x"),3), round(f(state,robot,"pos_base_y"),3),
                          "reward", reward, "moved", moved)
    finally:
        env.close()


if __name__ == "__main__":
    main()
