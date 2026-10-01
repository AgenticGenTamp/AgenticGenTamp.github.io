from env_client import make_env
import numpy as np, math
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
def trial(theta,arm_t,x0,label):
    env=make_env(); obs,info=env.reset(seed=42)
    def step(a):
        nonlocal obs
        obs,*_=env.step(np.array(a,dtype=float)); return d(obs,'robot')
    R=lambda: d(obs,'robot')
    def rot_to(th):
        for i in range(400):
            e=th-R()['theta']
            while e>math.pi: e-=2*math.pi
            while e<-math.pi: e+=2*math.pi
            if abs(e)<1e-7: break
            step([0,0,float(np.clip(e,-0.196,0.196)),0,0])
    def goto(axis,t):
        for i in range(900):
            v=R()['x'] if axis==0 else R()['y']; e=t-v
            if abs(e)<1e-7: break
            a=[0,0,0,0,0]; a[axis]=float(np.clip(e,-0.049,0.049)); step(a)
    def setarm(t):
        for mag in (0.05,0.005,0.0005):
            for i in range(400):
                e=t-R()['arm_joint']
                if abs(e)<1e-7: break
                p=R()['arm_joint']; step([0,0,0,float(np.clip(e,-mag,mag)),0])
                if abs(R()['arm_joint']-p)<1e-9: break
    goto(1,0.5); rot_to(0.0); setarm(arm_t); rot_to(theta); goto(0,x0)
    for mag in (0.049,0.005,0.0005,0.00005,0.00001):
        prev=None
        for i in range(1500):
            r=step([0,-mag,0,0,0])
            if prev is not None and abs(r['y']-prev)<1e-9: break
            prev=r['y']
    print("%-22s theta=%+.4f arm=%.3f x=%.4f MINY=%.6f"%(label,R()['theta'],R()['arm_joint'],R()['x'],R()['y']))
    env.close()
trial(math.pi/2,0.1,0.5,"arm UP retracted")
trial(math.pi/2,0.2,0.5,"arm UP extended")
trial(-math.pi/2,0.1,0.5,"arm DOWN retracted")
trial(0.0,0.2,0.5,"arm HORIZ extended")
trial(-math.pi/2,0.1,1.518,"arm DOWN far right")
trial(-math.pi/2,0.1,0.1,"arm DOWN far left")
