import numpy as np, sys
from calib_util import R, RD
import kin
seed=0
r=R(seed)
def yaw(q): w,x,y,z=q; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
r.gripper(1.0,10)
def moveto(pw,steps=80):
    q,e=r.ik_world(pw,RD); r.goto_q(q,maxsteps=steps,settle=4); return e
for (u,v) in [(0.0,-0.042),(0.0,0.0),(0.03,0.0)]:
    P=r.P('scoop_0'); th=yaw(r.Q('scoop_0'))
    c,s=np.cos(th),np.sin(th)
    pw=P[:2]+np.array([c*u-s*v,s*u+c*v])
    e=moveto([pw[0],pw[1],0.53]); print('above err',e, r.fk()[0].round(4), np.abs(r.q()-r.qi).max())
    q,e=r.ik_world([pw[0],pw[1],0.466],RD); print('ik err',e)
    for k in range(40):
        r.goto_q(q,maxsteps=1,settle=0)
        if k%4==0: print(k, r.fk()[0].round(4), 'lag',np.abs(r.q()-r.qi).max().round(4),'scoop',r.P('scoop_0').round(4),yaw(r.Q('scoop_0')).round(3))
    moveto([pw[0],pw[1],0.53])
