from env_client import make_env
import numpy as np, math

def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}

env=make_env()
obs,info=env.reset(seed=42)
names=[o.name for o in obs]
print("objects:",names)
print("robot0:",{k:round(v,5) for k,v in d(obs,'robot').items()})
for n in names:
    if n!='robot': print(n,{k:round(v,5) for k,v in d(obs,n).items()})

def step(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=float)); return d(obs,'robot')

def rot_to(th):
    for i in range(300):
        r=d(obs,'robot'); e=th-r['theta']
        while e>math.pi: e-=2*math.pi
        while e<-math.pi: e+=2*math.pi
        if abs(e)<1e-6: break
        step([0,0,np.clip(e,-0.196,0.196),0,0])
    return d(obs,'robot')['theta']

def push(axis,sign,coarse=0.049,fine=0.005):
    # axis 0=x,1=y
    for mag in (coarse,fine):
        prev=None
        for i in range(400):
            a=[0,0,0,0,0]; a[axis]=sign*mag
            r=step(a)
            v=r['x'] if axis==0 else r['y']
            if prev is not None and abs(v-prev)<1e-9: break
            prev=v
    r=d(obs,'robot'); return r['x'] if axis==0 else r['y']

def arm(sign):
    prev=None
    for mag in (0.099,0.01,0.002):
        for i in range(200):
            r=step([0,0,0,sign*mag,0])
            if prev is not None and abs(r['arm_joint']-prev)<1e-9: break
            prev=r['arm_joint']
    return d(obs,'robot')['arm_joint']

# --- Q2 arm limits in free space (go up high first)
th=rot_to(math.pi/2); print("theta set",round(th,5))
ymid=push(1,+1); print("max y (arm up retracted-ish) first push:",round(ymid,5))
print("arm min:",round(arm(-1),6))
print("arm max:",round(arm(+1),6))
print("state after arm max:",{k:round(v,5) for k,v in d(obs,'robot').items()})
print("arm back to min:",round(arm(-1),6))
env.close()
