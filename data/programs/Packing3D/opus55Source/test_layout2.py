import math
from layout import *
inner=(0.21,-0.14,0.39,0.14)
sq=local_footprint("cuboid",(0.05,0.05)); eq=local_footprint("tri",("equilateral",0.1,0.1)); rt=local_footprint("tri",("right",0.1,0.1))
h=0.0866
init=[(0.26,-0.09,0),(0.34,0.09,0),(0.39-h/3,-0.01,math.pi/2),(0.21+h/3,0.01,-math.pi/2)]
out,c,p=refine_layout([sq,sq,eq,eq],inner,init); print("2sq2eq",round(c,4),[tuple(round(v,4) for v in o) for o in p])
# right triangle: right angle at outer corner (0.39,0.04), legs along -x and -y: local legs +x,+y -> rotate 180
initr=[(0.26,-0.09,0),(0.34,0.09,0),(0.39-0.1/3,0.04-0.1/3,math.pi),(0.21+0.1/3,-0.04+0.1/3,0)]
out,c,p=refine_layout([sq,sq,rt,rt],inner,initr); print("2sq2rt",round(c,4),[tuple(round(v,4) for v in o) for o in p])
initm=[(0.26,-0.09,0),(0.34,0.09,0),(0.39-h/3,-0.01,math.pi/2),(0.21+0.1/3,-0.04+0.1/3,0)]
out,c,p=refine_layout([sq,sq,eq,rt],inner,initm); print("2sq eq rt",round(c,4),[tuple(round(v,4) for v in o) for o in p])
