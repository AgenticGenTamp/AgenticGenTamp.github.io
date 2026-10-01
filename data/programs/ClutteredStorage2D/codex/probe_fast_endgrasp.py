"""Test direct end-on grasps to skip released-block regrasp cycles."""
import math
from approach import GeneratedApproach
from env_client import make_env

env=make_env();s,info=env.reset(seed=0,options={"object_count":7})
p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info)
stored=set(p.to_clear)
direct=None
def g(o,k): return float(s.get(o,k))
for step in range(1000):
    old=p.stage; a=p.get_action(s)
    if old=="choose" and p.stage=="navigate" and not p.clear_mode and p.target_name not in stored:
        b=s.get_object_from_name(p.target_name); bx,by,bt=(g(b,k) for k in ("x","y","theta"))
        # End-on approach along the long axis, choosing the reachable side.
        opts=[bt,(bt+math.pi)%(2*math.pi)-math.pi]
        def cost(t):
            x,y=bx-.54*math.cos(t),by-.54*math.sin(t)
            return (max(0,.23-x)+max(0,x-4.77)+max(0,.23-y)+max(0,y-2.35))*100+math.hypot(x-g(s.get_object_from_name("robot"),"x"),y-g(s.get_object_from_name("robot"),"y"))
        th=min(opts,key=cost);p.approach_theta=th;p.goal_base=(bx-.54*math.cos(th),by-.54*math.sin(th));direct=p.target_name
        print("DIRECT",step+1,direct,round(th,2))
    if direct==p.target_name and p.stage in ("orient","retract") and not p.clear_mode:
        p.regrasped=True
    s,_,term,trunc,_=env.step(a)
    if step%100==99: print(step+1,p.stage,p.target_name)
    if term or trunc: print("END",step+1,term,trunc);break
env.close()
