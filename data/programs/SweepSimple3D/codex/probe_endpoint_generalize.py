"""Probe one-cube ballistic geometry across seeds."""
import math
import sys
import numpy as np
from env_client import make_env


def get(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def xy(s, n):
    fs = ("pos_base_x", "pos_base_y") if n == "robot" else ("x", "y")
    return np.array([get(s, n, fs[0]), get(s, n, fs[1])])


def run(seed, mode, offset=-0.30):
    env = make_env(); s, _ = env.reset(seed=seed, options={"object_count": 1})
    w0, c0 = xy(s, "wiper_0"), xy(s, "cube_0")
    qw, qz = get(s,"wiper_0","qw"), get(s,"wiper_0","qz")
    yaw = 2*math.atan2(qz, qw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    if mode == "localplus": target = w0 + .30*axis + np.array([offset, 0.])
    elif mode == "localminus": target = w0 - .29*axis + np.array([offset, 0.])
    else:
        direct=(c0-w0)/np.linalg.norm(c0-w0); th=math.radians(74)
        impact=np.array([math.cos(th)*direct[0]-math.sin(th)*direct[1],math.sin(th)*direct[0]+math.cos(th)*direct[1]])
        target=w0-.48*impact-.29*np.array([-impact[1],impact[0]])
    for _ in range(24):
        a=np.zeros(11,np.float32); a[:2]=np.clip(.8*(target-xy(s,"robot")),-.1,.1)
        s,*_=env.step(a)
    wdmax=cdmax=0.
    # Push world +x as suggested by endpoint-minus-x geometry.
    for _ in range(35):
        a=np.zeros(11,np.float32); a[0]=.035
        s,r,d,t,_=env.step(a)
        wdmax=max(wdmax,np.linalg.norm(xy(s,"wiper_0")-w0)); cdmax=max(cdmax,np.linalg.norm(xy(s,"cube_0")-c0))
    print(seed,mode,"yaw",round(yaw,3),"w",np.round(w0,3),"c",np.round(c0,3),"target",np.round(target,3),"wd",round(wdmax,3),"cd",round(cdmax,3),"delta",np.round(xy(s,"cube_0")-c0,3))
    env.close()


if __name__ == "__main__":
    run(int(sys.argv[1]),sys.argv[2],float(sys.argv[3]) if len(sys.argv)>3 else -.30)
