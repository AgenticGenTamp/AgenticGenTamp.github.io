from env_client import make_env
import numpy as np, math
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
env=make_env(); obs,info=env.reset(seed=42)
def step(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=float)); return d(obs,'robot')
def R(): return d(obs,'robot')
def rot_to(th):
    for i in range(400):
        e=th-R()['theta']
        while e>math.pi: e-=2*math.pi
        while e<-math.pi: e+=2*math.pi
        if abs(e)<1e-7: break
        step([0,0,float(np.clip(e,-0.196,0.196)),0,0])
def push(axis,sign,mags):
    for mag in mags:
        prev=None
        for i in range(1200):
            a=[0,0,0,0,0]; a[axis]=sign*mag
            r=step(a); v=r['x'] if axis==0 else r['y']
            if prev is not None and abs(v-prev)<1e-9: break
            prev=v
    r=R(); return round(r['x'] if axis==0 else r['y'],7)
def goto_x(t):
    for i in range(600):
        e=t-R()['x']
        if abs(e)<1e-7: break
        step([float(np.clip(e,-0.049,0.049)),0,0,0,0])
rot_to(math.pi/2)
for x in [0.2,0.5,0.9,1.1975,1.4]:
    push(1,-1,(0.049,)); goto_x(x)
    print("x=%.4f maxy(coarse .0005)=%s"%(R()['x'],push(1,+1,(0.049,0.0005))), end='  ')
    print("then .00005 =>",push(1,+1,(0.00005,)), " then .00001 =>", push(1,+1,(0.00001,)))
env.close()
