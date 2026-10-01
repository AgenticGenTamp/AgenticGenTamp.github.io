from sim import *
import numpy as np
s=S(1)
hx,hy,ht=s.pose('hook'); print('hook',hx,hy,ht)
d1=np.array([-np.cos(ht),-np.sin(ht)])
gp=np.array([hx,hy])+0.7*d1
th=ht+np.pi/2; u=np.array([np.cos(th),np.sin(th)])
for off in [0.6,0.5,0.45,0.4,0.35]:
    c=gp-off*u
    print('goto',off,s.goto(c[0],c[1],th), s.rob())
    for i in range(5): s.step([0,0,0,0.1,0])
    print(' ext',s.rob())
    for i in range(12): s.step([0,0,0,0,-0.02])
    print(' close',s.rob(),'held',s.g('hook','held'), s.pose('hook'))
    if s.g('hook','held')>0: break
    for i in range(12): s.step([0,0,0,0,0.02])
    for i in range(5): s.step([0,0,0,-0.1,0])
