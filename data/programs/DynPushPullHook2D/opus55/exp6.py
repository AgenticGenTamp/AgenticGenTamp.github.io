from sim import *
import numpy as np
from env_client import make_env
env=make_env()
for lat in [0.0,0.053,-0.053]:
  for ins in [0.05,0.12,0.18]:
    s=S(1,env)
    hx,hy,ht=s.pose('hook')
    d1=np.array([-np.cos(ht),-np.sin(ht)]); n=np.array([np.sin(ht),-np.cos(ht)])
    end=np.array([hx,hy])+1.4*d1+lat*n
    u=-d1; th=np.arctan2(u[1],u[0])
    s.goto(s.g('robot','x'),s.g('robot','y'),th)
    c=end-0.5*u; s.goto(c[0],c[1],th)
    c=end+ins*u-0.37*u; s.goto(c[0],c[1],th)
    moved=s.pose('hook')
    for i in range(12): s.step([0,0,0,0,-0.02])
    held=s.g('hook','held'); gap=s.g('robot','finger_gap')
    for i in range(10): s.step([0,-0.03,0,0,0])
    print(lat,ins,'held',held,'gap',round(gap,3),'hook before',moved,'after',s.pose('hook'))
