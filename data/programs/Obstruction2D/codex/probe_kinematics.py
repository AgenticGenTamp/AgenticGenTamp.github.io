import math
import numpy as np
from env_client import make_env


def vals(s, obj, fs):
    return {f: float(s.get(obj, f)) for f in fs}


def summarize(seed):
    e = make_env()
    s, info = e.reset(seed=seed)
    rob = s.get_objects(e.observation_space.get_type("crv_robot"))[0]
    block = s.get_objects(e.observation_space.get_type("target_block"))[0]
    surf = s.get_objects(e.observation_space.get_type("target_surface"))[0]
    rect = s.get_objects(e.observation_space.get_type("rectangle"))
    print("SEED", seed, "names", s.get_object_names())
    print("robot", vals(s, rob, ["x","y","theta","base_radius","arm_joint","arm_length","vacuum","gripper_height","gripper_width"]))
    print("block", vals(s, block, ["x","y","theta","width","height"]))
    print("surface", vals(s, surf, ["x","y","theta","width","height"]))
    print("rects", [(o.name, vals(s,o,["x","y","theta","width","height","static"])) for o in rect])
    e.close()


def one_steps(seed):
    acts = [
        np.array([.05,0,0,0,0],np.float32), np.array([0,.05,0,0,0],np.float32),
        np.array([0,0,.19634954,0,0],np.float32), np.array([0,0,0,.1,0],np.float32),
        np.array([0,0,0,-.1,0],np.float32), np.array([0,0,0,0,1],np.float32)]
    for a in acts:
        e=make_env(); s,_=e.reset(seed=seed)
        r=s.get_objects(e.observation_space.get_type("crv_robot"))[0]
        b=s.get_objects(e.observation_space.get_type("target_block"))[0]
        before=vals(s,r,["x","y","theta","arm_joint","arm_length","vacuum"])
        bb=vals(s,b,["x","y","theta"])
        s,re,te,tr,info=e.step(a)
        after=vals(s,r,["x","y","theta","arm_joint","arm_length","vacuum"])
        ba=vals(s,b,["x","y","theta"])
        print("ONE",a.tolist(),"R",before,"=>",after,"B",bb,"=>",ba,"rew",re,te,tr,info)
        e.close()


if __name__ == "__main__":
    for seed in range(3): summarize(seed)
    one_steps(0)
