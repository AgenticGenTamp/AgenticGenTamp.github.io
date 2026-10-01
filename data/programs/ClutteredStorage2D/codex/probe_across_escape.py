"""Probe a high-lane detour while carrying a shelf payload."""
import sys
from approach import GeneratedApproach
from env_client import make_env

env=make_env();s,info=env.reset(seed=int(sys.argv[1]),options={"object_count":7})
p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info)
escape=False;last_x=None;stall=0
def g(o,k):return float(s.get(o,k))
for step in range(1000):
    r=s.get_object_from_name("robot");a=p.get_action(s)
    if (p.stage=="across_transport" or escape) and p.target_name:
        b=s.get_object_from_name(p.target_name)
        if not escape:
            if last_x is not None and abs(g(b,"x")-last_x)<.001:stall+=1
            else:stall=0
            last_x=g(b,"x")
            if stall>=3:escape=True
        if escape:
            ex=p.preplace[0]-g(b,"x")
            if abs(ex)>.01:
                a=p._action(dy=2.22-g(r,"y"),vac=1) if g(r,"y")<2.20 else p._action(dx=ex,vac=1)
            else:
                p.stage="insert";p.last_insert_y=g(b,"y");p.insert_stall=0;p.insert_toggle=0
                escape=False;a=p._action(vac=1)
    s,_,term,trunc,_=env.step(a)
    if step%100==99:
        b=s.get_object_from_name(p.target_name)
        print(step+1,p.stage,p.target_name,"r",round(g(r,"x"),2),round(g(r,"y"),2),"b",round(g(b,"x"),2),round(g(b,"y"),2))
    if term or trunc:print("END",step+1,term,trunc);break
env.close()
