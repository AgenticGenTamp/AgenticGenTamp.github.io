import numpy as np, sys
from env_client import make_env
seed=1
def run(approach_offset, push_dir, label):
    env=make_env(); obs,info=env.reset(seed=seed)
    names=[n for n in sorted(obs.get_object_names()) if n!='robot']
    c=obs.get_object_from_name(names[0]); cp=np.array([float(obs.get(c,'x')),float(obs.get(c,'y'))])
    def base(o):
        r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y']])
    # first move to staging point around chair (radius 1.2), avoiding chair by going around
    stage = cp + approach_offset
    for phase,tgt in [(0,cp+approach_offset*2.0),(1,stage),(2,cp+push_dir*0.1)]:
        for i in range(200):
            b=base(obs); d=tgt-b
            if np.linalg.norm(d)<0.06: break
            a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.1,0.1)
            obs,rew,term,trunc,info=env.step(a)
            if term or trunc:
                cc=obs.get_object_from_name(names[0])
                print(label,"phase",phase,"TERM",round(rew,4),"robot",np.round(base(obs),3),"chair",round(float(obs.get(cc,'x')),3),round(float(obs.get(cc,'y')),3))
                env.close(); return
    cc=obs.get_object_from_name(names[0])
    print(label,"no term; chair",round(float(obs.get(cc,'x')),3),round(float(obs.get(cc,'y')),3))
    env.close()
for ang in [0,90,180,270]:
    th=np.radians(ang); off=np.array([np.cos(th),np.sin(th)])*1.2
    run(off, -off/1.2, f"approach_from_{ang}")
