"""Probe controls after the near-target seed-1 pile transfer."""

import sys, numpy as np
from approach import GeneratedApproach
from env_client import make_env

def centroid(s):
    ps=[]
    for n in s.get_object_names():
        if n.startswith("cube_"):
            o=s.get_object_from_name(n); ps.append([s.get(o,"x"),s.get(o,"y")])
    return np.mean(ps,axis=0)

def main(dim,sign):
    e=make_env();s,info=e.reset(seed=1);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
    for _ in range(300):s=e.step(p.get_action(s))[0]
    c0=centroid(s)
    for _ in range(20):
        a=np.zeros(11,np.float32);a[10]=0;a[dim]=sign*.1;s,r,d,tr,_=e.step(a)
    print(dim,sign,np.round(centroid(s)-c0,4),r,d);e.close()
if __name__=="__main__":main(int(sys.argv[1]),float(sys.argv[2]))
