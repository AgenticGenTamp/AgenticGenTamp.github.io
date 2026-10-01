"""Sequential cardinal rake legs with fixed-target crossing stops (seed 0)."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

COLORS=("red","green","blue","yellow")

def xy(s,n):
    o=s.get_object_from_name(n)
    return np.array([float(s.get(o,"x")),float(s.get(o,"y"))])

def leg(env,s,info,name,kind,target,axis=None,direction=0):
    p=GeneratedApproach(env.action_space,env.observation_space,{})
    p.reset(s,info); p.plan=[(name,kind)]
    # Keep fixed initial target even if a receptacle was displaced.
    color=COLORS[(int(name[4:])-1)%4]; p.bin_targets[color]=target.copy()
    for k in range(275):
        s,r,t,tr,info=env.step(p.get_action(s))
        if axis is not None and k>190:
            v=xy(s,name)[axis]
            if (direction>0 and v>=target[axis]) or (direction<0 and v<=target[axis]):
                print("STOP",name,kind,k,np.round(xy(s,name),4)); break
        if t or tr: break
    return s,info,r,t,tr

def main():
    env=make_env(); s,info=env.reset(seed=0,options={"object_count":4})
    targets={c:xy(s,"bin_"+c).copy() for c in COLORS}
    # Red's calibrated diagonal arc; a blue-x clearing leg then exposes green.
    sequence=[("cube1","first",None,0),
              ("cube3","first",0,1),
              ("cube2","first",1,1),("cube2","second",0,-1),
              ("cube3","first",0,1),("cube3","second",1,1),
              ("cube4","first",1,-1),("cube4","second",0,1)]
    r=-1.;t=tr=False
    for i,(name,kind,axis,direction) in enumerate(sequence):
        color=COLORS[(int(name[4:])-1)%4]
        s,info,r,t,tr=leg(env,s,info,name,kind,targets[color],axis,direction)
        print("LEG",i,name,kind,"pos",np.round(xy(s,name),4),"d",
              round(float(np.linalg.norm(xy(s,name)-targets[color])),4),"r",r)
        if t or tr: break
    ds={n:round(float(np.linalg.norm(xy(s,n)-targets[COLORS[(int(n[4:])-1)%4]])),6)
        for n in ("cube1","cube2","cube3","cube4")}
    print("FINAL",ds,"reward",r,"terminated",t,"truncated",tr)
    env.close()

if __name__=="__main__": main()
