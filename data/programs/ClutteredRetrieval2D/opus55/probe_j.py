from probe_lib import *
from geom import rect_corners, circ_poly
o,_=env.reset(seed=11)
gx,gy,_=bcenter(o,'target_region')
o,ok=goto(o,x=gx-0.4,y=gy); print('near region',ok,rob(o))
o,ok=goto(o,x=gx,y=gy); print('at region center',ok,rob(o))
o,ok=goto(o,x=gx+0.3,y=gy); print('past region',ok,rob(o))
# lateral suction extent: block seed 11, gripper offset laterally
