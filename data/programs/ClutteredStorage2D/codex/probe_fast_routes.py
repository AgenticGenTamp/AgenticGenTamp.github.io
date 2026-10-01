"""Probe diagonal payload transport as a step-saving shortcut."""
from approach import GeneratedApproach
from env_client import make_env

env=make_env();s,info=env.reset(seed=0,options={"object_count":7})
p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info)
def g(o,k): return float(s.get(o,k))
for step in range(1100):
    a=p.get_action(s)
    if p.stage in ("lower_transport","across_transport") and p.target_name:
        b=s.get_object_from_name(p.target_name)
        ex,ey=p.preplace[0]-g(b,"x"),2.25-g(b,"y")
        if max(abs(ex),abs(ey))<.01:
            p.stage="preplace";a=p._action(vac=1)
        else:
            a=p._action(dx=ex,dy=ey,vac=1)
    s,_,term,trunc,_=env.step(a)
    if step%100==99: print(step+1,p.stage,p.target_name)
    if term or trunc: print("END",step+1,term,trunc);break
env.close()
