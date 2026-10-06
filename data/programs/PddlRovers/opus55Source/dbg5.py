import numpy as np, math, sys
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.ss_pybullet import *
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=int(sys.argv[1]))
b = env.current_backend
r=int(sys.argv[2]); x=float(sys.argv[3]); y=float(sys.argv[4])
print([ (round(get_point(o)[0],2), round(get_point(o)[1],2)) for o in b._objectives] if hasattr(b,'_objectives') else [k for k in vars(b) if 'obj' in k])
with b.client():
    rover=b._rovers[r]; joints=b._base_joints[r]
    for th in np.linspace(-math.pi, math.pi, 25):
        set_joint_positions(rover, joints, (x,y,th))
        vis=[int(b.image_visible(r,o)) if True else 0 for o in b._objectives]
        print(round(th,2), vis, int(b.com_visible(r)))
