"""Coarse online search for a table-height grasp configuration."""
from env_client import make_env
import numpy as np


def v(s, name, feat):
    return float(s.get(s.get_object_from_name(name), feat))


env = make_env(); s, info = env.reset(seed=0)
bx = v(s,"target_block","pose_x"); by=v(s,"target_block","pose_y")
xs = [-.02,-.12,-.22,-.32,-.42,-.52,-.62]
configs = [(q2,q4) for q2 in (-.35,-.15,.05,.25,.45,.65)
                    for q4 in np.arange(-2.5,1.51,.2)]
for ci,(q2,q4) in enumerate(configs):
    sweep = xs if ci%2 == 0 else xs[::-1]
    for tx in sweep:
        a=np.zeros(11,np.float32)
        a[0]=np.clip(tx-v(s,"robot","pos_base_x"),-.2,.2)
        a[1]=np.clip(by-v(s,"robot","pos_base_y"),-.2,.2)
        a[4]=np.clip(q2-v(s,"robot","joint_2"),-.2,.2)
        a[6]=np.clip(q4-v(s,"robot","joint_4"),-.2,.2)
        a[10]=-1
        s,r,term,trunc,inf=env.step(a)
        if v(s,"robot","grasp_active"):
            held=[n for n in s.get_object_names() if n!="robot" and v(s,n,"grasp_active")]
            print("FOUND",ci,tx,q2,q4,"base",v(s,"robot","pos_base_x"),v(s,"robot","pos_base_y"),"q",[v(s,"robot",f"joint_{i}") for i in range(1,8)],"held",held,"tf",[v(s,"robot",f"grasp_tf_{c}") for c in "xyz"])
            env.close(); raise SystemExit
        if trunc:
            print("TRUNC",ci); env.close(); raise SystemExit
print("NONE",len(configs)*len(xs)); env.close()
