from sim import *
import numpy as np
def hookgeo(s):
    hx,hy,ht=s.pose('hook'); u=np.array([np.cos(ht),np.sin(ht)]); n=np.array([np.sin(ht),-np.cos(ht)])
    end=np.array([hx,hy])-1.4*u+0.053*n
    return end,u,ht
def grasp(s):
    end,u,ht=hookgeo(s)
    s.goto(s.g('robot','x'),s.g('robot','y'),ht)
    c=end-0.5*u; s.goto(c[0],c[1],ht)
    end,u,ht=hookgeo(s)
    c=end+0.12*u-0.37*u; s.goto(c[0],c[1],ht)
    for i in range(12): s.step([0,0,0,0,-0.02])
    return s.g('hook','held')
