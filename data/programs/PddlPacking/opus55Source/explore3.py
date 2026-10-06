import numpy as np, time
from simcore import *
s = Sim(4)
st = s.get_state()
q0 = np.array(st[:10], float)
def coll(q):
    s.set_state(st); s.set_conf(q)
    with s.env.client():
        return s.env._in_collision()
for bx in np.arange(-0.7,-0.3,0.02):
    q=q0.copy(); q[0]=bx
    if coll(q): print('front collide at bx', bx); break
# from +x side facing -x
for bx in np.arange(0.9,0.3,-0.02):
    q=q0.copy(); q[0]=bx; q[2]=np.pi
    if coll(q): print('back side collide at bx', bx); break
for by in np.arange(-1.2,-0.5,0.02):
    q=q0.copy(); q[0]=0; q[1]=by; q[2]=np.pi/2
    if coll(q): print('side collide at by', by); break
t=time.time()
for i in range(100): coll(q0)
print('coll time', (time.time()-t)/100)
t=time.time()
for i in range(100): s.fk(q0)
print('fk time', (time.time()-t)/100)
