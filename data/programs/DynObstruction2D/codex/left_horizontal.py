import math
import numpy as np
from env_client import make_env

for seed in (11, 14, 30, 35, 43):
    e=make_env(); s,_=e.reset(seed=seed); typ=e.observation_space.get_type
    r=s.get_objects(typ('kin_robot'))[0]; b=s.get_objects(typ('target_block'))[0]; q=s.get_objects(typ('target_surface'))[0]
    g=lambda o,f:float(s.get(o,f))
    # Repeat short horizontal left pushes from a safely staged overhead pose.
    done=False
    for cycle in range(8):
        for _ in range(30): s,*_=e.step(np.array([0,np.clip(1.1-g(r,'y'),-.049,.049),0,-.099,0],float))
        tx=min(2.76,g(b,'x')+.43)
        for _ in range(70):
            err=(math.pi-g(r,'theta')+math.pi)%(2*math.pi)-math.pi
            s,*_=e.step(np.array([np.clip(tx-g(r,'x'),-.049,.049),0,np.clip(err,-.19,.19),-.099,0],float))
        ty=max(.24,g(b,'y')-.08)
        for _ in range(35): s,*_=e.step(np.array([0,np.clip(ty-g(r,'y'),-.049,.049),0,-.099,0],float))
        for _ in range(11): s,*_=e.step(np.array([0,0,0,0,-.019],float))
        for _ in range(70):
            s,_,done,tr,_=e.step(np.array([-.004,0,0,0,0],float))
            if done or tr: break
        if done: break
    print(seed,done,'block',round(g(b,'x'),2),round(g(b,'y'),2),'surface',round(g(q,'x'),2),flush=True)
    e.close()
