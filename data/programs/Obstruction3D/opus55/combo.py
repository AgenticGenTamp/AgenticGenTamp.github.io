import numpy as np
from th import H
from kin import ik, down_R
h=H(3)
n='target_block'; p=h.pose(n)[:3]; he=h.he(n); top=p[2]+he[2]
h.goto((p[0],p[1],top+0.06),0.0)
# final step: move down to top+0.028 with close in same action
q=h.q(); qt,e=ik(h.base(),q,np.array([p[0],p[1],top+0.028]),down_R(0.0))
a=np.zeros(11); a[3:10]=np.clip(qt-q,-.2,.2); a[10]=-1; h.step(a)
print('tool',h.tool()[:3,3].round(3),'grasped',h.grasped())
h.grip(-1); print('after extra close', h.grasped())
# move to release spot then combined open
tgt=np.array([p[0]+0.1,p[1],top+0.028+0.0015])
h.goto(tgt+np.array([0,0,0.03]),0.0)
q=h.q(); qt,e=ik(h.base(),q,tgt,down_R(0.0))
a=np.zeros(11); a[3:10]=np.clip(qt-q,-.2,.2); a[10]=1; h.step(a)
print('tool',h.tool()[:3,3].round(3),'grasped after combined open',h.grasped(), h.pose(n)[:3].round(3))
