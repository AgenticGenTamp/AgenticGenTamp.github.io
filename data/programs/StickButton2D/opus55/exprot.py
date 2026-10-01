import sys, math, numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
def rd(obs):
    R=S=None; B=[]
    for n in obs.get_object_names():
        o=obs.get_object_from_name(n); t=o.type.name
        if t=='crv_robot': R=o
        elif t=='rectangle': S=o
        elif t=='circle': B.append(o)
    g=obs.get
    r=dict((f,g(R,f)) for f in ('x','y','theta','arm_joint','vacuum'))
    s=tuple(g(S,f) for f in ('x','y','theta'))
    b=[(o.name,g(o,'x'),g(o,'y'),g(o,'color_r'),g(o,'color_g')) for o in B]
    return r,s,b
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
def to_grasp(seed, cnt=None):
    obs,info=env.reset(seed=seed, options=({'object_count':cnt} if cnt else None))
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    acts=[]; k=None
    for t in range(600):
        a=ap.get_action(obs); acts.append(a); obs,*_=env.step(a)
        if ap.grasped and k is None: k=t
        if k is not None and t>=k+3: break
    return obs, acts
def replay(seed, acts, cnt=None):
    obs,_=env.reset(seed=seed, options=({'object_count':cnt} if cnt else None))
    for a in acts: obs,*_=env.step(a)
    return obs
def step(obs,dx=0,dy=0,dth=0,da=0):
    o2,*_=env.step(np.array([dx,dy,dth,da,1.0],dtype=np.float32)); return o2
def pred(r,s,dth):
    px,py=r['x'],r['y']; c,sn=math.cos(dth),math.sin(dth)
    ax,ay=s[0]-px,s[1]-py
    return (px+c*ax-sn*ay, py+sn*ax+c*ay, wrap(s[2]+dth))
