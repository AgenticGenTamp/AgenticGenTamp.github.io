from env_client import make_env
import numpy as np, math
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
env=make_env()
obs,info=env.reset(seed=42)
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
    return R()['theta']
def push(axis,sign,mags=(0.049,0.005)):
    for mag in mags:
        prev=None
        for i in range(500):
            a=[0,0,0,0,0]; a[axis]=sign*mag
            r=step(a); v=r['x'] if axis==0 else r['y']
            if prev is not None and abs(v-prev)<1e-9: break
            prev=v
    r=R(); return r['x'] if axis==0 else r['y']
def arm(sign,mags=(0.099,0.01,0.002)):
    for mag in mags:
        prev=None
        for i in range(300):
            r=step([0,0,0,sign*mag,0])
            if prev is not None and abs(r['arm_joint']-prev)<1e-9: break
            prev=r['arm_joint']
    return R()['arm_joint']

# free space: theta=0 (arm horizontal to the right), go to mid-height
rot_to(0.0)
print("theta0 pos",round(R()['x'],4),round(R()['y'],4))
# move to middle-ish y
for i in range(10): step([0,-0.049,0,0,0])
print("y now",round(R()['y'],4))
print("ARM MIN(free horiz):",round(arm(-1),6))
print("ARM MAX(free horiz):",round(arm(+1),6))
amax=R()['arm_joint']
arm(-1); amin=R()['arm_joint']
print("amin,amax",amin,amax)

# Q1/Q4: theta=+pi/2 arm retracted
rot_to(math.pi/2)
print("== theta=+pi/2, arm retracted (%.3f) =="%R()['arm_joint'])
print(" max y:",round(push(1,+1),5))
print(" min x:",round(push(0,-1),5))
print(" max x:",round(push(0,+1),5))
print(" min y (from here):",round(push(1,-1),5))
print(" state:",{k:round(v,4) for k,v in R().items() if k in('x','y','theta','arm_joint')})
env.close()
