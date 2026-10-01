import numpy as np
from scoop_util import *
r=R(0); r.gripper(1.0,10)
pw,_=local2world(r,-0.09,0)
lin(r,[pw[0],pw[1],0.50]); print('at 0.50', r.fk()[0].round(4), 'qi-q',(r.qi-r.q()).round(3))
lin(r,[pw[0],pw[1],0.43],maxsteps=30); print('push', r.fk()[0].round(4), 'qi-q',(r.qi-r.q()).round(3))
lin(r,[pw[0],pw[1],0.50]); print('back 0.50', r.fk()[0].round(4), 'qi-q',(r.qi-r.q()).round(3))
for k in range(20): r.step(dq=np.zeros(7))
print('wait', r.fk()[0].round(4), 'qi-q',(r.qi-r.q()).round(3))
