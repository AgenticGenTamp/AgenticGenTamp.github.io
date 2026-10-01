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
        for i in range(1500):
            a=[0,0,0,0,0]; a[axis]=sign*mag
            r=step(a); v=r['x'] if axis==0 else r['y']
            if prev is not None and abs(v-prev)<1e-9: break
            prev=v
    r=R(); return round(r['x'] if axis==0 else r['y'],7)
def goto(axis,t):
    for i in range(800):
        v=R()['x'] if axis==0 else R()['y']
        e=t-v
        if abs(e)<1e-7: break
        a=[0,0,0,0,0]; a[axis]=float(np.clip(e,-0.049,0.049)); step(a)
    return round(R()['x'] if axis==0 else R()['y'],5)
rot_to(math.pi/2)
goto(1,0.6)
for x in [0.15,0.3,0.6,0.9,1.1975,1.35,1.5,1.517]:
    goto(1,0.6); got=goto(0,x)
    my=push(1,+1,(0.049,0.005,0.0005,0.00005,0.00001))
    goto(1,0.6)
    print("x=%.5f maxy=%s"%(got,my))
env.close()
