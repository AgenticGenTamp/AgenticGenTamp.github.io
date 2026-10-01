import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, rotz, DOWN, GRASP_V, ik, HOME, fk
for seed in [25,65,79]:
    env=make_env(); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space,env.observation_space,{})
    base=ap._base(obs)
    print("seed",seed,"count",info.get("object_count"))
    for name,p,he in ap._cubes(obs):
        pb=ap._to_base(p,base)
        print("  ",name,"world",np.round(p,3),"base",np.round(pb,3),"dist %.3f"%np.hypot(pb[0],pb[1]))
        for yaw in [0.0,0.35,-0.35]:
            R=rotz(yaw)@DOWN
            bp=pb-R@GRASP_V
            for dz in [0.30,0.12,0.0]:
                q,e=ik(bp+np.array([0,0,dz]),R,[HOME]+[HOME+np.random.default_rng(1).normal(0,1,7) for _ in range(4)])
                print("     yaw %.2f dz %.2f ikerr %.4f"%(yaw,dz,e))
    env.close()
