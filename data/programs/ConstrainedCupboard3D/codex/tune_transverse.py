"""Read-only comparison of push side/cutoff variants on transverse rods."""
import concurrent.futures
import math
import numpy as np
from env_client import make_env

SOURCE = open("approach.py", encoding="utf-8").read()

def wrap(x):
    return (x + math.pi) % (2*math.pi) - math.pi

def run(seed, side_scale, cutoff):
    # Alter only dynamic side magnitude and depth cutoff in an in-memory copy.
    src = SOURCE.replace("side = .10 if ry < self.slot_y[index] else -.10",
                         f"side = {side_scale} if ry < self.slot_y[index] else -{side_scale}")
    src = src.replace("if rx >= 1.75:", f"if rx >= {cutoff}:")
    ns = {}; exec(src, ns)
    env = make_env(); s, info = env.reset(seed=seed, options={"object_count": 1})
    pol = ns["GeneratedApproach"](env.action_space, env.observation_space, {})
    pol.reset(s, info); rod = pol.rods[0]
    p0 = np.array([s.get(rod,k) for k in ("x","y","z")], float)
    yaw0 = wrap(2*math.atan2(float(s.get(rod,"qz")), float(s.get(rod,"qw"))))
    for t in range(env.max_steps):
        s,r,term,trunc,info=env.step(pol.get_action(s))
        if term or trunc: break
    p1=np.array([s.get(rod,k) for k in ("x","y","z")], float)
    yaw1=wrap(2*math.atan2(float(s.get(rod,"qz")), float(s.get(rod,"qw"))))
    dist=math.hypot(p1[0]-2.,p1[1]-pol.slot_y[0])
    env.close()
    return seed, side_scale, cutoff, yaw0, yaw1, p0, p1, dist

if __name__ == "__main__":
    seeds=range(12)
    # Screen initial orientation with resets, then test variants only on yaw~0.
    transverse=[]
    for seed in seeds:
        env=make_env(); s,_=env.reset(seed=seed,options={"object_count":1})
        typ=env.observation_space.get_type("mujoco_movable_object")
        rod=list(s.get_objects(typ))[0]
        yaw=wrap(2*math.atan2(float(s.get(rod,"qz")),float(s.get(rod,"qw"))))
        env.close()
        if abs(yaw)<.5 or abs(abs(yaw)-math.pi)<.5: transverse.append(seed)
    print("TRANSVERSE",transverse,flush=True)
    jobs=[(sd,side,cut) for sd in transverse[:2]
          for side in (.04,.07,.10) for cut in (1.65,1.80)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        out=list(ex.map(lambda x:run(*x),jobs))
    for x in out:
        sd,side,cut,y0,y1,p0,p1,d=x
        print(sd,side,cut,"yaw",round(y0,2),round(y1,2),"p",np.round(p1,3),"d",round(d,3),flush=True)
