from env_client import make_env


def val(state, obj, feature):
    try:
        return float(state.get(obj, feature))
    except Exception:
        return None


def main():
    for count in (1, 5, 10):
        for seed in range(5):
            env = make_env()
            obs, info = env.reset(seed=seed, options={"object_count": count})
            mov = obs.get_objects(env.observation_space.get_type("mujoco_movable_object"))
            fix = obs.get_objects(env.observation_space.get_type("mujoco_fixture"))
            robots = obs.get_objects(env.observation_space.get_type("mujoco_tidybot_robot"))
            print("CASE", count, seed, "info", repr(info), "names", [o.name for o in mov])
            for o in mov:
                fs = [val(obs, o, k) for k in ("x", "y", "z", "bb_x", "bb_y", "bb_z")]
                print(" M", o.name, *(round(x, 4) for x in fs))
            for o in fix:
                fs = [val(obs, o, k) for k in ("x", "y", "z", "qw", "qz")]
                print(" F", o.name, *(round(x, 4) for x in fs))
            for o in robots:
                fs = [val(obs, o, k) for k in ("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper")]
                print(" R", o.name, *(round(x, 4) for x in fs))
            env.close()


if __name__ == "__main__":
    main()
