import numpy as np
from env_client import make_env
from scoop_lib import Scoop
PI=np.pi

def mk(seed=0):
    env=make_env(); s=Scoop(env,seed); s.fetch_hook(); s.rel(); return env,s

def panth(s): return -PI/2-s.rt

def load_pan(s, sx=1.45, sweep_x=0.55, press=12):
    """lower pan on the left, sweep left, ending with objects in the pan. Robot ends low."""
    s.rel(); ty=s.pan_y()
    s.goto(y=2.30,th=panth(s),gap=0.08)
    s.goto(x=1.66,y=2.30,th=panth(s),gap=0.08)
    s.goto(x=sx,y=2.30,th=panth(s))
    s.goto(x=sx,y=ty,th=panth(s))
    s.goto(x=sweep_x,y=ty,th=panth(s))
    for _ in range(press): s.step([-0.03,0,0,0,0])
    return s.in_pan()

def objinfo(s):
    return [(round(s.g('x',o),3),round(s.g('y',o),3)) for o in s.smalls()]
