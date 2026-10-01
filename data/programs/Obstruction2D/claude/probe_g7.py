from env_client import make_env
import numpy as np, math
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
def run(rotate_to=None):
    env=make_env(); obs,info=env.reset(seed=42)
    def step(a):
        nonlocal obs
        obs,*_=env.step(np.array(a,dtype=float)); return d(obs,'robot')
    R=lambda: d(obs,'robot')
    if rotate_to is not None:
        for i in range(400):
            e=rotate_to-R()['theta']
            while e>math.pi: e-=2*math.pi
            while e<-math.pi: e+=2*math.pi
            if abs(e)<1e-7: break
            step([0,0,float(np.clip(e,-0.196,0.196)),0,0])
    for mag in (0.049,0.005,0.0005,0.00005):
        prev=None
        for i in range(1500):
            r=step([0,mag,0,0,0])
            if prev is not None and abs(r['y']-prev)<1e-9: break
            prev=r['y']
    print("theta=%.4f arm=%.3f x=%.5f MAXY=%.7f"%(R()['theta'],R()['arm_joint'],R()['x'],R()['y']))
    env.close()
run(None)
run(math.pi/2)
run(0.0)
run(math.pi)
