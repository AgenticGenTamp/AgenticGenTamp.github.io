import sys; sys.path.insert(0,'scratch')
from gm_lib import *
grip=float(sys.argv[1])
r=R(27,(-0.47,0.22,0.0))
if grip<0: print("opening after close",r.grip(-1.0))
bc,byaw=block(r.o)
def descend(xy,yaw,label):
    assert r.move_tool([xy[0],xy[1],0.95],yaw)
    z=0.95
    for st in (0.01,0.002,0.0005):
        while r.try_tool([xy[0],xy[1],z-st],yaw): z-=st
    print(label,"min z",round(z,4),flush=True)
    r.move_tool([xy[0],xy[1],0.95],yaw)
descend(bc[:2],byaw,"block centre yaw=b")
descend(bc[:2],byaw+math.pi/4,"block centre yaw=b+45")
descend([-0.1,0.45],byaw,"empty table")
