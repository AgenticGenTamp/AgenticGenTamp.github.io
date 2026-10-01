"""Probe robust broad-face pickup on the first loose-block attempt."""
import math
from approach import GeneratedApproach
from env_client import make_env

env=make_env();s,info=env.reset(seed=0,options={"object_count":7})
p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info); stored=set(p.to_clear)
p.attempts.update({"block3":1,"block6":1})
def g(o,k):return float(s.get(o,k))
for step in range(1100):
    old=p.stage;a=p.get_action(s)
    if False and old=="choose" and p.stage=="navigate" and not p.clear_mode and p.target_name not in stored:
        b=s.get_object_from_name(p.target_name);bx,by,bt=(g(b,k) for k in ("x","y","theta"));r=s.get_object_from_name("robot")
        opts=[bt+math.pi/2,bt-math.pi/2]
        def cost(t):
            x,y=bx-.54*math.cos(t),by-.54*math.sin(t)
            return (max(0,.22-x)+max(0,x-4.78)+max(0,.22-y)+max(0,y-2.3))*50+math.hypot(x-g(r,"x"),y-g(r,"y"))
        th=min(opts,key=cost);p.approach_theta=(th+math.pi)%(2*math.pi)-math.pi;p.goal_base=(bx-.54*math.cos(th),by-.54*math.sin(th))
        print("BROAD",step+1,p.target_name)
    s,_,term,trunc,_=env.step(a)
    if step%100==99:
        b=s.get_object_from_name(p.target_name) if p.target_name else None
        print(step+1,p.stage,p.target_name,[] if b is None else [round(g(b,k),3) for k in ("x","y","theta")])
    if term or trunc:print("END",step+1,term,trunc);break
env.close()
