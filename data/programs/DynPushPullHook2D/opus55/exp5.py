from sim import *
import numpy as np
s=S(1)
hx,hy,ht=s.pose('hook'); print('hook',hx,hy,ht)
d1=np.array([-np.cos(ht),-np.sin(ht)])
gp=np.array([hx,hy])+0.7*d1
th=ht+np.pi/2; u=np.array([np.cos(th),np.sin(th)])
for off in [0.6,0.5,0.45,0.4,0.37]:
    c=gp-off*u; s.goto(c[0],c[1],th)
    print(off, s.rob(), s.pose('hook'))
for i in range(12): s.step([0,0,0,0,-0.02])
print(' close',s.rob(),'held',s.g('hook','held'), s.pose('hook'))
for i in range(10): s.step([0.03,0,0,0,0]); 
print('move', s.rob(), s.pose('hook'))
for i in range(10): s.step([0,0,0.06,0,0]); 
print('rot', s.rob(), s.pose('hook'))
for i in range(20): s.step([0,0.05,0,0,0]); 
print('up', s.rob(), s.pose('hook'))
