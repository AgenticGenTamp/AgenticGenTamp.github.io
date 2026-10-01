import sys
from grip_util import *
g=G(0)
H=[]
orig=g.step
def st(db=(0,0,0),dq=None):
    r=orig(db,dq); rb=g.rob()
    H.append((g.g, g.obs.get(rb,'pos_gripper'), g.obs.get(rb,'vel_gripper'), g.tool().round(4), g.P('cube_17').round(4)))
    return r
g.step=st
for i in range(3):
    H.clear()
    ok,info=pick(g,'cube_17',verbose=True)
    for h in H[-75:]:
        if h[0]>0.5 or h[3][2]<0.5: print(h)
    if ok: break
print('placed',place(g,'cube_17',(0.35,-0.28)).round(4))
