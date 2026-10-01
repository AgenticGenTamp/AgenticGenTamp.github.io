import sys; sys.path.insert(0,'scratch')
from gm_lib import *
r=R(27,(-0.47,0.22,0.0))
bc,byaw=block(r.o); print("block",bc.round(4),round(byaw,4),"grip",rstate(r.o)[10])
# check IK accuracy
r.move_tool([bc[0]-0.2,bc[1],1.0],byaw)
p,Rm=fk(r.base,r.cfg()); print("tool",p.round(4),"R",Rm.round(3))
