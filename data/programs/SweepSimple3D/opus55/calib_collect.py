"""Grasp a cube, then move through random arm configs + base poses; log (base,q,cube pose)."""
from env_client import make_env
import numpy as np, sys, kin, json
from rutil import Bot
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
name=sys.argv[2] if len(sys.argv)>2 else 'cube_1'
ang=float(sys.argv[3]) if len(sys.argv)>3 else np.pi/2
D=float(sys.argv[4]) if len(sys.argv)>4 else 0.55
rng=np.random.default_rng(seed)
env=make_env(); obs,_=env.reset(seed=0); b=Bot(env,obs)
c=b.obj(name,('x','y','z','qw','qx','qy','qz'))
yaw=2*np.arctan2(c[6],c[3])
base=np.array([c[0]+D*np.cos(ang),c[1]+D*np.sin(ang),ang+np.pi])
b.goto(base_t=base,grip=0.0)
bp=b.base(); print('base target',np.round(base,3),'actual',np.round(bp,3))
def reach(p):
    q,ok,e=kin.ik(bp,b.q(),p,'down',yaw=yaw); b.goto(q_t=q,tol=0.002,max_steps=300)
reach(c[:3]+[0,0,0.12]); reach(c[:3]+[0,0,0.0]); print('pre-close cube',np.round(b.obj(name),4),'fk',np.round(b.tip()[0],4)); b.wait(15,grip=1.0)
reach(c[:3]+[0,0,0.1]); print('lift cube',np.round(b.obj(name),4),'fk',np.round(b.tip()[0],4))
reach(c[:3]+[0,0,0.3])
qlift=b.q().copy()
data=[]
for i in range(18):
    if i%6==5:
        bt=b.base()+np.array([rng.uniform(-.2,.2),rng.uniform(-.2,.2),rng.uniform(-.8,.8)])
        b.goto(base_t=bt,tol=0.003,max_steps=100)
    qt=qlift+rng.uniform(-0.5,0.5,7)*np.array([1,.5,1,.6,1,.6,1])
    b.goto(q_t=qt,tol=0.001,max_steps=200)
    b.wait(3)
    cp=b.obj(name,('x','y','z','qw','qx','qy','qz'))
    p,R=b.tip()
    print(i,'cube',np.round(cp[:3],3),'fk',np.round(p,3),'d',np.round(cp[:3]-p,3))
    if cp[2]>0.05: data.append(dict(base=b.base().tolist(),q=b.q().tolist(),cube=cp.tolist()))
json.dump(data,open(f'calib_data_{seed}.json','w'))
for k in range(4):
    b.wait(10); print('hold',np.round(b.obj(name)-b.tip()[0],4))
