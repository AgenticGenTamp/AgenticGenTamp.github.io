import time
from layout import *
inner=(0.21,-0.14,0.39,0.14)
sq=local_footprint("cuboid",(0.05,0.05)); eq=local_footprint("tri",("equilateral",0.1,0.1)); rt=local_footprint("tri",("right",0.1,0.1))
for name,fps in [("2sq+2eq",[sq,sq,eq,eq]),("2sq+eq+rt",[sq,sq,eq,rt]),("2sq+2rt",[sq,sq,rt,rt]),("sq+2eq",[sq,eq,eq])]:
    t=time.time(); out,c=optimize_layout(fps,inner); print(name, round(c,4), round(time.time()-t,2), [tuple(round(v,3) for v in o) for o in out])
