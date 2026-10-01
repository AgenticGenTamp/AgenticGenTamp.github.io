from env_client import make_env
import numpy as np, sys, kin
from rutil import Bot
env=make_env(); obs,_=env.reset(seed=0); b=Bot(env,obs)
name=sys.argv[1] if len(sys.argv)>1 else 'cube_1'
c=b.obj(name,('x','y','z','qw','qx','qy','qz'))
yaw=2*np.arctan2(c[6],c[3])
base=np.array([c[0],c[1]+0.55,-np.pi/2])
b.goto(base_t=base,grip=0.0)
bp=b.base(); print('base',np.round(bp,3))
def reach(p,steps=300):
    q,ok,e=kin.ik(bp,b.q(),p,'down',yaw=yaw)
    b.goto(q_t=q,tol=0.003,max_steps=steps)
    print('target',np.round(p,3),'ik ok',ok,'fk',np.round(b.tip()[0],3),'steps',b.nsteps)
reach(c[:3]+[0,0,0.12])
reach(c[:3]+[0,0,0.0])
print('cube before close',np.round(b.obj(name),3))
b.wait(15,grip=1.0)
print('cube after close',np.round(b.obj(name),3),'fk',np.round(b.tip()[0],3))
reach(c[:3]+[0,0,0.15])
b.wait(5)
print('cube lifted',np.round(b.obj(name),3),'fk',np.round(b.tip()[0],3),'grip',b.gripper())
