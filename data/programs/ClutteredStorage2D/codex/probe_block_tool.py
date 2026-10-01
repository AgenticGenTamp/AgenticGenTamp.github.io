"""Use the first carried vertical block as a tool to nudge block0 right."""
import numpy as np
from approach import GeneratedApproach
from env_client import make_env

env=make_env(); s,info=env.reset(seed=0,options={"object_count":7})
p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info)
def g(o,k): return float(s.get(o,k))
phase=None;ticks=0
for step in range(1000):
    a=p.get_action(s)
    b0=s.get_object_from_name("block0")
    if p.target_name and p.target_name!="block0":
        t=s.get_object_from_name(p.target_name)
        if phase is None and p.stage=="insert" and g(t,"y")>2.34:
            phase="left";ticks=0;print("INTERVENE",step,p.target_name,g(t,"x"),g(t,"y"))
        if phase:
            ticks+=1
            if phase=="left":
                a=p._action(dx=.24-g(t,"x"),vac=1)
                if g(t,"x")<.255: phase="rotate";ticks=0
            elif phase=="rotate":
                # Diagonal high-force ram around the left lip.
                a=p._action(dx=-.05,dy=.05,da=.1,vac=1)
                if ticks>12: phase="up";ticks=0
            elif phase=="up":
                a=p._action(dy=.03,vac=1)
                if g(t,"y")>2.62 or ticks>15: phase="right";ticks=0
            elif phase=="right":
                a=p._action(dx=.04,vac=1)
                if ticks>8: phase="done"
            else: a=p._action(vac=0)
    s,_,term,trunc,_=env.step(a)
    if phase and (ticks%2==0 or term):
        t=s.get_object_from_name(p.target_name)
        print(step+1,phase,"tool",round(g(t,"x"),3),round(g(t,"y"),3),
              "b0",round(g(b0,"x"),3),round(g(b0,"y"),3),round(g(b0,"theta"),3),term)
    if term or trunc: print("END",step+1,term);break
env.close()
