import numpy as np, sys
from scoop_util import *
r=R(0); print('grip init',r.grip()); r.gripper(1.0,20); print('closed air',r.grip()); r.gripper(0.0,20); print('open',r.grip())
# cube grasp
c=r.P('cube_0'); print('cube',c)
