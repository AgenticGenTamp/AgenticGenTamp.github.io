from env_client import make_env


def g(s, o, *fs):
    return [round(s.get(o, f), 3) for f in fs]


for seed in range(10):
    env = make_env()
    s, info = env.reset(seed=seed)
    typ = env.observation_space.get_type
    rob = s.get_objects(typ("kin_robot"))[0]
    blk = s.get_objects(typ("target_block"))[0]
    surf = s.get_objects(typ("target_surface"))[0]
    obs = [o for o in s.get_objects(typ("dyn_rectangle")) if o.name.startswith("obstruction")]
    print(seed, info, "R", g(s, rob, "x", "y", "theta", "arm_joint", "arm_length", "finger_gap"),
          "B", g(s, blk, "x", "y", "width", "height"),
          "S", g(s, surf, "x", "y", "width", "height"),
          "O", [(o.name, g(s, o, "x", "y", "width", "height")) for o in obs])
    env.close()
