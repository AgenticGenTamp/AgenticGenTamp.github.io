import math
import numpy as np
from env_client import make_env


def trial(seed, fwd, lat):
    env=make_env(); s,_=env.reset(seed=seed); sp=env.observation_space
    R=s.get_objects(sp.get_type("kin_robot"))[0]; B=s.get_objects(sp.get_type("target_block"))[0]
    th=s.get(R,"theta"); bx=s.get(B,"x"); by=s.get(B,"y")
    # Desired base makes block lie at local gripper coordinate (fwd, lat).
    tx=bx-fwd*math.cos(th)+lat*math.sin(th); ty=by-fwd*math.sin(th)-lat*math.cos(th)
    try:
        for _ in range(30):
            dx=tx-s.get(R,"x"); dy=ty-s.get(R,"y")
            if abs(dx)<.004 and abs(dy)<.004: break
            s,*_=env.step(np.array([max(-.049,min(.049,dx)),max(-.049,min(.049,dy)),0,0,0],np.float32))
        moved=(s.get(B,"x")-bx,s.get(B,"y")-by)
        hit=None
        for i in range(12):
            s,*_=env.step(np.array([0,0,0,0,-.02],np.float32))
            if s.get(B,"held")>.5: hit=i+1; break
        return hit,moved,(s.get(B,"x"),s.get(B,"y")),s.get(R,"finger_gap")
    except Exception as e:
        return "ERR",(0,0),(0,0),str(e)
    finally:
        env.close()


for f in (.55,.70,.85,1.0):
    for lat in (-.3,-.15,0,.15,.3):
        print(f,lat,trial(0,f,lat),flush=True)
