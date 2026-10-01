from env_client import make_env
import numpy as np, math
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
env=make_env(); obs,info=env.reset(seed=42)
def step(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=float)); return d(obs,'robot')
R=lambda: d(obs,'robot')
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
        for i in range(1500):
            a=[0,0,0,0,0]; a[axis]=sign*mag
            r=step(a); v=r['x'] if axis==0 else r['y']
            if prev is not None and abs(v-prev)<1e-9: break
            prev=v
    r=R(); return round(r['x'] if axis==0 else r['y'],6)
def setarm(target):
    for mag in (0.05,0.005,0.0005):
        prev=None
        for i in range(400):
            e=target-R()['arm_joint']
            if abs(e)<1e-7: break
            r=step([0,0,0,float(np.clip(e,-mag,mag)),0])
            if prev is not None and abs(r['arm_joint']-prev)<1e-9: break
            prev=r['arm_joint']
    return round(R()['arm_joint'],5)
def goto(axis,t):
    for i in range(900):
        v=R()['x'] if axis==0 else R()['y']
        e=t-v
        if abs(e)<1e-7: break
        a=[0,0,0,0,0]; a[axis]=float(np.clip(e,-0.049,0.049)); step(a)
    return round(R()['x'] if axis==0 else R()['y'],5)

# work in open left area
rot_to(0.0); goto(1,0.5); goto(0,0.6)
print("--- max y vs arm_joint, theta=+pi/2 (arm up)")
for a in [0.1,0.125,0.15,0.175,0.2]:
    goto(1,0.5); rot_to(0.0); print("  set arm",setarm(a),end='')
    rot_to(math.pi/2); print("  maxy=",push(1,+1))
print("--- min y vs arm_joint, theta=-pi/2 (arm down), x=0.6")
for a in [0.1,0.15,0.2]:
    goto(1,0.5); rot_to(0.0); setarm(a); rot_to(-math.pi/2)
    print("  arm=%.3f miny=%s"%(R()['arm_joint'],push(1,-1)))
print("--- min y at several x, theta=-pi/2, arm=0.1")
rot_to(0.0); goto(1,0.5); setarm(0.1); rot_to(-math.pi/2)
for x in [0.1,0.3,0.6,0.9,1.518]:
    goto(1,0.5); gx=goto(0,x); print("  x=%.4f miny=%s"%(gx,push(1,-1)))
print("--- max x with arm horizontal extended")
goto(1,0.5); rot_to(0.0); setarm(0.2)
print("  theta=0 arm=%.3f maxx=%s"%(R()['arm_joint'],push(0,+1)))
rot_to(math.pi); print("  theta=pi arm=%.3f minx=%s"%(R()['arm_joint'],push(0,-1)))
print("--- max y theta=0 arm=0.2 (horizontal near ceiling)")
goto(0,0.6); print("  maxy=",push(1,+1))
env.close()
