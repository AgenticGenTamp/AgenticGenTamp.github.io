import sys
from grip_util import *
g=G(0)
for sp in [(0.35,-0.28),(0.35,-0.10),(0.33,-0.19),(0.62,-0.10),(0.62,-0.30)]:
    b=np.array([-0.15,sp[1],0]); q,e=g.ik_world(np.array([sp[0],sp[1],0.47]),base=b); print('reach',sp,round(e,5))
ok,info=pick(g,'cube_17',verbose=True)
print('placed',place(g,'cube_17',(0.35,-0.28)).round(4), cyaw(g.Q('cube_17')))
for i in range(3):
    ok,info=pick(g,'cube_17',verbose=True)
    if ok: print('placed',place(g,'cube_17',(0.35,-0.28)).round(4))
print('steps',len(g.rew))
