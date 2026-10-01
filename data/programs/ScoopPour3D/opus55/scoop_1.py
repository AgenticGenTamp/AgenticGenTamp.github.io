import numpy as np, sys
from calib_util import R, RD
import kin
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
r=R(seed); o=r.obs
c=o.get_object_from_name('cube_0'); print('cube bb',[o.get(c,f) for f in ['bb_x','bb_y','bb_z']])
def yaw(q): w,x,y,z=q; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
r.gripper(1.0,10)
P0=r.P('scoop_0'); th=yaw(r.Q('scoop_0')); print('scoop',P0,th)
def moveto(pw,steps=60):
    q,e=r.ik_world(pw,RD); r.goto_q(q,maxsteps=steps,settle=4); return e
res=[]
for u in np.arange(-0.07,0.0701,0.014):
  for v in np.arange(-0.042,0.0421,0.014):
    P=r.P('scoop_0'); th=yaw(r.Q('scoop_0'))
    c,s=np.cos(th),np.sin(th)
    pw=P[:2]+np.array([c*u-s*v,s*u+c*v])
    moveto([pw[0],pw[1],0.53])
    moveto([pw[0],pw[1],0.466],steps=40)
    for k in range(4): r.step(dq=np.zeros(7))
    z=r.fk()[0][2]; P2=r.P('scoop_0'); th2=yaw(r.Q('scoop_0'))
    d=np.linalg.norm(P2-P)
    print('u %.3f v %.3f  tipz %.4f  scoop moved %.4f dth %.3f scoopz %.4f'%(u,v,z-0.001,d,th2-th,P2[2]),flush=True)
    moveto([pw[0],pw[1],0.53])
print('final scoop',r.P('scoop_0'),r.Q('scoop_0'))
