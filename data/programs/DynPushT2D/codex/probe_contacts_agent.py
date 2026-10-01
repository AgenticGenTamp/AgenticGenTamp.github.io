"""Contact / push probes on a near-axis-aligned central T block."""
import numpy as np
from env_client import make_env


def move(env, obs, target, order=(0, 1), step=.04):
    obs = obs.copy()
    for axis in order:
        while abs(float(target[axis] - obs[16 + axis])) > 1e-4:
            a = np.zeros(2, np.float32)
            a[axis] = np.clip(target[axis] - obs[16 + axis], -step, step)
            obs, *_ = env.step(a)
    return obs


def probe(name, start_offset, inward, order, pre=None):
    env=make_env(); o,_=env.reset(seed=81); initial=o.copy(); center=o[:2].copy()
    if pre is not None:
        o=move(env,o,np.array(pre),order=(1,0))
    target=center+np.array(start_offset)
    o=move(env,o,target,order=order)
    at=o.copy(); first=None; contact_r=None; hist=[]
    transit_delta=at[:3]-initial[:3]
    for k in range(60):
        before=o.copy(); o,*rest=env.step(np.array(inward,np.float32)*.02)
        bd=float(np.linalg.norm(o[:2]-initial[:2]))
        if first is None and (bd>1e-5 or abs(float(o[2]-initial[2]))>1e-5):
            first=k; contact_r=float(np.linalg.norm(before[16:18]-before[:2]))
        if first is not None and k in (first,first+1,first+4,first+9,59):
            hist.append((k,np.round(o[:3]-initial[:3],4).tolist(),np.round(o[16:18],4).tolist()))
    # settle ten steps
    pushed=o.copy()
    for _ in range(10): o,*_=env.step(np.zeros(2,np.float32))
    print(name,"theta",round(float(initial[2]),3),"at",np.round(at[16:18],3).tolist(),
          "transit",np.round(transit_delta,4).tolist(),
          "first",first,"rad",None if contact_r is None else round(contact_r,3),
          "push",np.round(pushed[:3]-initial[:3],4).tolist(),
          "settle_extra",np.round(o[:3]-pushed[:3],5).tolist(),"hist",hist)
    env.close()


# safe axis order / peripheral waypoint keeps transit from crossing the object.
probe("from-left",(-1.,0),(1,0),(1,0))
probe("from-right",(1.,0),(-1,0),(1,0),pre=(3.831,.7))
probe("from-bottom",(0,-1.4),(0,1),(1,0),pre=(2.831,.7))
probe("from-top",(0,1),(0,-1),(1,0))
