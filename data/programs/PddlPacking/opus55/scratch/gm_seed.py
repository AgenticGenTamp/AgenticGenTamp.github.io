from env_client import make_env
import numpy as np, math
env=make_env(); T=env.observation_space.get_type
for s in range(40):
    o,info=env.reset(seed=s)
    bl=[]
    for b in o.get_objects(T("block")):
        g=lambda k: float(o.get(b,k))
        bl.append((round(g("pose_x"),3),round(g("pose_y"),3),round(g("pose_z"),3),round(2*math.atan2(g("pose_qz"),g("pose_qw")),3)))
    pl=[(round(float(o.get(p,"pose_x")),3),round(float(o.get(p,"pose_y")),3),round(float(o.get(p,"half_extent_x")),3),round(float(o.get(p,"half_extent_y")),3)) for p in o.get_objects(T("surface"))]
    print(s,bl,pl)
