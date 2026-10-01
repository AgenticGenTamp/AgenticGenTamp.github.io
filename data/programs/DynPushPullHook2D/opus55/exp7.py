from sim import *
import numpy as np
s=S(1)
def hookgeo():
    hx,hy,ht=s.pose('hook'); u=np.array([np.cos(ht),np.sin(ht)]); n=np.array([np.sin(ht),-np.cos(ht)])
    end=np.array([hx,hy])-1.4*u+0.053*n
    return end,u,ht
end,u,ht=hookgeo()
s.goto(s.g('robot','x'),s.g('robot','y'),ht)
for k in range(60):
    end,u,ht=hookgeo()
    c=end-0.5*u; 
    rx,ry,rt=s.g('robot','x'),s.g('robot','y'),s.g('robot','theta')
    if np.hypot(c[0]-rx,c[1]-ry)<0.005: break
    s.step([c[0]-rx,c[1]-ry,(ht-rt+np.pi)%(2*np.pi)-np.pi,0,0])
for k in range(20):
    end,u,ht=hookgeo(); c=end+0.12*u-0.37*u
    rx,ry,rt=s.g('robot','x'),s.g('robot','y'),s.g('robot','theta')
    s.step([c[0]-rx,c[1]-ry,(ht-rt+np.pi)%(2*np.pi)-np.pi,0,0])
for i in range(12): s.step([0,0,0,0,-0.02])
print('held',s.g('hook','held'),s.rob(),s.pose('hook'),s.t)
s.goto(1.75,0.35,None); print(s.rob(),s.pose('hook'))
s.goto(1.75,0.35,np.pi/2); print('rot',s.rob(),s.pose('hook'))
for i in range(30): s.step([0,0.05,0,0,0])
print('up',s.rob(),s.pose('hook'))
for i in range(5): s.step([0,0,0,0.1,0])
print('ext',s.rob(),s.pose('hook'))
s.goto(1.75,1.0,None)
for i in range(5): s.step([0,0,0,0.1,0])
print('ext',s.rob(),s.pose('hook'))
for i in range(40): s.step([0.05,0,0,0,0])
print('right',s.rob(),s.pose('hook'))
for i in range(80): s.step([-0.05,0,0,0,0])
print('left',s.rob(),s.pose('hook'))
