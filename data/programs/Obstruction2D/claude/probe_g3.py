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
def push(axis,sign,mags=(0.049,0.005,0.0005)):
    for mag in mags:
        prev=None
        for i in range(600):
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

res={}
# ---- theta=-pi/2 (arm down), arm retracted
rot_to(-math.pi/2); setarm(-1)
print("A) theta=-pi/2 arm=%.3f: maxy=%s minx=%s maxx=%s"%(R()['arm_joint'],push(1,+1),push(0,-1),push(0,+1)))
print("   miny at maxx:",push(1,-1))
# ---- theta=+pi/2 arm retracted, refined
rot_to(math.pi/2)
print("B) theta=+pi/2 arm=%.3f: maxy=%s minx=%s maxx=%s"%(R()['arm_joint'],push(1,+1),push(0,-1),push(0,+1)))
print("   miny at maxx (arm up):",push(1,-1))
# arm up EXTENDED max y
setarm(+1)
print("C) theta=+pi/2 arm=0.2: maxy=",push(1,+1))
setarm(-1)
# ---- Q3: theta=-pi/2, arm at MIN, min y at several x
rot_to(-math.pi/2); setarm(-1)
push(1,+1)  # go high to travel freely
push(0,-1)  # go far left
xs=[]
for target in [0.11,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.52]:
    # travel at high y
    push(1,+1)
    for i in range(400):
        e=target-R()['x']
        if abs(e)<1e-6: break
        step([float(np.clip(e,-0.049,0.049)),0,0,0,0])
    miny=push(1,-1)
    xs.append((round(R()['x'],4),miny))
    print("   x=%.4f  min y(arm down,min)=%s"%xs[-1])
# ---- Q3b: arm at MAX pointing down, min y
push(1,+1); setarm(+1)
for target in [0.3,0.8,1.3]:
    push(1,+1)
    for i in range(400):
        e=target-R()['x']
        if abs(e)<1e-6: break
        step([float(np.clip(e,-0.049,0.049)),0,0,0,0])
    print("   ARMMAX x=%.4f min y=%s arm=%.3f"%(R()['x'],push(1,-1),R()['arm_joint']))
env.close()
