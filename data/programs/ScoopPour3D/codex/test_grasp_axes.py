"""Map controls while holding a verified scoop grasp."""

import math, sys
import numpy as np
from env_client import make_env


def read(s,n,fs):
    o=s.get_object_from_name(n); return np.array([s.get(o,f) for f in fs])


def main(dim,sign,steps=10):
    env=make_env(); s,_=env.reset(seed=0)
    sp=read(s,"scoop_0",("x","y","z","qw","qx","qy","qz")); q=sp[3:]
    yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    base=read(s,"robot",("pos_base_x","pos_base_y")); target=base+.06*np.array([math.cos(yaw),math.sin(yaw)])
    for _ in range(12):
        b=read(s,"robot",("pos_base_x","pos_base_y")); a=np.zeros(11,np.float32); a[:2]=np.clip(.8*(target-b),-.1,.1); a[10]=1; s=env.step(a)[0]
    for i in range(32):
        a=np.zeros(11,np.float32); a[4]=.1; a[6]=.06; a[10]=0 if i>=30 else 1; s=env.step(a)[0]
    for _ in range(8): a=np.zeros(11,np.float32); s=env.step(a)[0]
    for _ in range(18):
        a=np.zeros(11,np.float32); a[4]=-.1;a[6]=-.06;s=env.step(a)[0]
    before=read(s,"scoop_0",("x","y","z"))
    for _ in range(steps):
        a=np.zeros(11,np.float32);a[dim]=sign*.1;s,r,d,tr,_=env.step(a)
    print(dim,sign,np.round(read(s,"scoop_0",("x","y","z"))-before,4),"pos",np.round(read(s,"scoop_0",("x","y","z")),4),r)
    env.close()


if __name__=="__main__":main(int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 10)
