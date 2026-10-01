import numpy as np
from env_client import make_env
from approach import world_chain
from farroute import FarRoute, robot_state
env=make_env()
for seed in (0,1,2,3):
    obs,info=env.reset(seed=seed); pol=FarRoute(); pl=pol.reset(obs,info)
    for tag,q in (("pre",pl["q_pre"]),("grasp",pl["path"][-1])):
        pts,_=world_chain(q, pl["base"])
        s=" ".join("(%.2f,%.2f,%.2f)"%tuple(p) for p in pts)
        print(seed,pl["name"],tag,"base_x %.3f"%pl["base"][0],s)
env.close()
