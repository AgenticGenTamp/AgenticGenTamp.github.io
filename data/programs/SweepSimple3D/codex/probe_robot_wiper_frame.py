import sys
import numpy as np
from env_client import make_env

def get(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n):
    fs=("pos_base_x","pos_base_y") if n=="robot" else ("x","y")
    return np.array([get(s,n,fs[0]),get(s,n,fs[1])])

def run(seed,a=.108,b=.55):
    e=make_env(); s,_=e.reset(seed=seed,options={"object_count":1})
    r0,w0,c0=xy(s,"robot"),xy(s,"wiper_0"),xy(s,"cube_0")
    u=(r0-w0); u/=np.linalg.norm(u); right=np.array([u[1],-u[0]])
    target=w0-a*u-b*right
    md=0.
    for _ in range(24):
        act=np.zeros(11,np.float32); act[:2]=np.clip(.8*(target-xy(s,"robot")),-.1,.1)
        s,r,d,t,_=e.step(act); md=max(md,float(np.linalg.norm(xy(s,"cube_0")-c0)))
    print(seed,"r",np.round(r0,3),"w",np.round(w0,3),"c",np.round(c0,3),"t",np.round(target,3),"max",round(md,4),"dc",np.round(xy(s,"cube_0")-c0,4),"dw",np.round(xy(s,"wiper_0")-w0,4),flush=True)
    e.close()

run(int(sys.argv[1]),*(map(float,sys.argv[2:])))
