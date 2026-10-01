import numpy as np
from env_client import make_env
from ctl import act, rget, oget
from grasp import wrap
env=make_env()
def scan(obs, env, xs):
    """descend at each x, report contact height (hook top)."""
    res={}
    def H(): return np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])
    for tx in xs:
        # rotate arm up, retract, close
        for _ in range(80):
            d=wrap(np.pi/2-rget(obs,'theta'))
            if abs(d)<0.01: break
            obs,_,_,_,_=env.step(act(dth=d,da=-0.1,dg=-0.02))
        for _ in range(400):
            dx=np.clip(tx-rget(obs,'x'),-0.05,0.05); dy=np.clip(1.45-rget(obs,'y'),-0.05,0.05)
            if abs(dx)<0.003 and abs(dy)<0.003: break
            obs,_,_,_,_=env.step(act(dx=dx,dy=dy))
        h0=H(); hit=None
        for _ in range(300):
            obs,_,_,_,_=env.step(act(dy=-0.005))
            if np.abs(H()-h0).max()>1e-5: hit=rget(obs,'y'); break
        res[tx]=None if hit is None else round(hit-0.27,4)
        if hit is not None:
            for _ in range(10): obs,_,_,_,_=env.step(act(dy=0.05))
    return obs,res
obs,info=env.reset(seed=42)
def step(a):
    global obs
    obs,_,_,_,_=env.step(a)
hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
stand=np.array([hx,hy])+(-0.107-0.275)*u-0.45*nv
for _ in range(60): step(act(dth=wrap(np.pi+hth-rget(obs,'theta')),da=-0.1,dg=-0.02))
tgt=stand-0.3*u
for _ in range(300):
    dx=np.clip(tgt[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(tgt[1]-rget(obs,'y'),-0.05,0.05)
    if abs(dx)<0.004 and abs(dy)<0.004: break
    step(act(dx=dx,dy=dy))
for _ in range(12): step(act(dx=0.05))
hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
print("hook",round(hx,3),round(hy,3),round(hth,4))
for _ in range(12): step(act(dx=-0.05))
u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
xs=[2.0,2.4,2.8]
obs,res=scan(obs,env,xs)
for tx in xs:
    # model: bar top surface at this x: solve along bar
    t=(tx-hx)/u[0]  # local x coordinate approx (ignoring n contribution)
    pred=hy+t*u[1]
    print("x",tx,"measured top",res[tx],"model top(origin line)",round(pred,4))
env.close()
