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
def push(axis,sign,mags=(0.049,0.005,0.0005,0.00005)):
    for mag in mags:
        prev=None
        for i in range(800):
            a=[0,0,0,0,0]; a[axis]=sign*mag
            r=step(a); v=r['x'] if axis==0 else r['y']
            if prev is not None and abs(v-prev)<1e-9: break
            prev=v
    r=R(); return round(r['x'] if axis==0 else r['y'],6)
def setarm(sign):
    for mag in (0.099,0.005):
        prev=None
        for i in range(300):
            r=step([0,0,0,sign*mag,0])
            if prev is not None and abs(r['arm_joint']-prev)<1e-9: break
            prev=r['arm_joint']
    return round(R()['arm_joint'],5)
def goto_x(t):
    for i in range(500):
        e=t-R()['x']
        if abs(e)<1e-6: break
        step([float(np.clip(e,-0.049,0.049)),0,0,0,0])

# go to left open region x~0.5
rot_to(math.pi/2); setarm(-1); push(1,+1); goto_x(0.5)
print("MAXY arm=0.1 up :",push(1,+1))
setarm(+1); print("MAXY arm=0.2 up :",push(1,+1))
setarm(-1)
rot_to(-math.pi/2); print("MAXY arm=0.1 down:",push(1,+1))
setarm(+1); print("MAXY arm=0.2 down:",push(1,+1))
setarm(-1)
print("MINY arm=0.1 down:",push(1,-1))
rot_to(math.pi/2); print("MINY arm=0.1 up  :",push(1,-1))
setarm(+1); print("MINY arm=0.2 up  :",push(1,-1))
setarm(-1)
# x walls with theta=0 (arm +x) and theta=pi (arm -x)
push(1,+1)
for th,lab in [(0.0,'theta=0'),(math.pi,'theta=pi'),(math.pi/2,'theta=pi/2')]:
    rot_to(th); setarm(+1)
    print("%s arm=0.2: minx=%s maxx=%s"%(lab,push(0,-1),push(0,+1)))
    setarm(-1)
    print("%s arm=0.1: minx=%s maxx=%s"%(lab,push(0,-1),push(0,+1)))
env.close()
