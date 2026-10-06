import time
from layout import *
S=("cuboid",(0.05,0.05)); E=("tri",("equilateral",0.1,0.1)); R=("tri",("right",0.1,0.1))
for name,ps in [("E",[E]),("R",[R]),("SE",[S,E]),("SR",[S,R]),("SEE",[S,E,E]),("SER",[S,E,R]),("SRE",[S,R,E]),("SRR",[S,R,R]),("SSEE",[S,S,E,E]),("SSER",[S,S,E,R]),("SSRE",[S,S,R,E]),("SSRR",[S,S,R,R])]:
    t=time.time(); out,c=compute_layout(ps); print(name, round(c,4), round(time.time()-t,2))
