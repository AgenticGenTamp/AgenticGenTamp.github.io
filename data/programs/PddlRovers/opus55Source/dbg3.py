import numpy as np, math
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.ss_pybullet import *
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=3)
b = env.current_backend
with b.client():
    rover=b._rovers[1]; joints=b._base_joints[1]
    pil=15; aabb=get_aabb(pil); c=np.array([(aabb[0][0]+aabb[1][0])/2,(aabb[0][1]+aabb[1][1])/2])
    print("pillar c", c, get_pose(pil))
    worst=0
    res={}
    for th in np.linspace(-math.pi, math.pi, 25):
        for ang in np.linspace(-math.pi, math.pi, 37):
            # find min distance free along direction ang
            lo=0.1; 
            for d in np.arange(0.45,0.15,-0.005):
                set_joint_positions(rover, joints, (c[0]+d*math.cos(ang), c[1]+d*math.sin(ang), th))
                if pairwise_collision(rover, pil): lo=d; break
            # rel angle in rover frame
            rel=round(((ang+math.pi - th)+math.pi)%(2*math.pi)-math.pi,2)
            worst=max(worst,lo)
            res[(round(th,2),round(ang,2))]=lo
    print("worst", worst)
    # dist profile vs relative angle (direction from rover to pillar = ang+pi, in rover frame)
    prof={}
    for (th,ang),lo in res.items():
        rel=((ang+math.pi-th)+math.pi)%(2*math.pi)-math.pi
        k=round(rel/ (math.pi/12))
        prof[k]=max(prof.get(k,0),lo)
    for k in sorted(prof): print(round(k*15), round(prof[k],3))
