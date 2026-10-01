import sys
from env_client import make_env

for seed in [int(x) for x in sys.argv[1:]]:
    e = make_env(); s, info = e.reset(seed=seed); T=e.observation_space.get_type
    print("seed", seed, info)
    for tn in ("kin_robot","target_block","target_surface","dyn_rectangle"):
        for o in s.get_objects(T(tn)):
            fs = ["x","y","theta","width","height"] if tn != "kin_robot" else ["x","y","theta","arm_joint","finger_gap"]
            print(tn, o.name, {f:round(float(s.get(o,f)),3) for f in fs})
    e.close()
