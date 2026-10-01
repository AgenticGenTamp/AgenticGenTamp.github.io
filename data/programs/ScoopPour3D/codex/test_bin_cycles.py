"""Repeat the one-sided arm stroke that nudges the loaded tray targetward."""

import numpy as np
from approach import GeneratedApproach
from env_client import make_env


def xyz(s, n):
    o=s.get_object_from_name(n); return np.array([s.get(o,f) for f in ("x","y","z")])


def main():
    env=make_env(); s,info=env.reset(seed=0)
    p=GeneratedApproach(env.action_space,env.observation_space,{ }); p.reset(s,info)
    for _ in range(67): s=env.step(p.get_action(s))[0]
    for _ in range(12):
        a=np.zeros(11,np.float32); a[10]=1; a[:2]=.035*p.source_dir; a[2]=-.1
        s=env.step(a)[0]
    p0=xyz(s,"bin_yellow_0")
    robot=s.get_object_from_name("robot")
    b0=np.array([s.get(robot,"pos_base_x"),s.get(robot,"pos_base_y")])
    qfs=["pos_arm_joint%d"%i for i in range(1,8)]
    q0=np.array([s.get(robot,f) for f in qfs])
    for cycle in range(3):
        for _ in range(20):
            a=np.zeros(11,np.float32); a[10]=1; a[5]=-.1; a[6]=.1
            s,r,done,trunc,_=env.step(a)
        print(cycle,np.round(xyz(s,"bin_yellow_0")-p0,4),r,done)
        for _ in range(30):
            q=np.array([s.get(robot,f) for f in qfs])
            a=np.zeros(11,np.float32); a[10]=1
            a[3:10]=np.clip(.7*(q0-q),-.1,.1)
            s=env.step(a)[0]
        # Follow the displaced tray so the next identical stroke contacts it.
        for _ in range(8):
            shift=xyz(s,"bin_yellow_0")[:2]-p0[:2]
            base=np.array([s.get(robot,"pos_base_x"),s.get(robot,"pos_base_y")])
            a=np.zeros(11,np.float32); a[10]=1
            a[:2]=np.clip(.7*(b0+shift-base),-.1,.1)
            s=env.step(a)[0]
    env.close()


if __name__=="__main__": main()
