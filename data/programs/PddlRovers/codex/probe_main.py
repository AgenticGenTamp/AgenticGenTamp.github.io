import numpy as np
from env_client import make_env

SEL = [(-1 + (k + .5) / 3) for k in range(6)]


def xy(s, name):
    o = s.get_object_from_name(name)
    return (float(s.get(o, "x")), float(s.get(o, "y")))


def run(seed, commands):
    e = make_env(); s, _ = e.reset(seed=seed)
    print("start", xy(s,"rover0"), xy(s,"rover1"))
    for n,(dx0,dy0,dx1,dy1) in enumerate(commands):
        a=np.zeros(8,np.float32); a[[0,1,4,5]]=[dx0,dy0,dx1,dy1]
        s,r,t,tr,i=e.step(a)
        print(n+1,xy(s,"rover0"),xy(s,"rover1"),t,i)
    e.close()


run(0, [(-.2,0,0,0)]*8 + [(0,.2,0,0)]*24)
