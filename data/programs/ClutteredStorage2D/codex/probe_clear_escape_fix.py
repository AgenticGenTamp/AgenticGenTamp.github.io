"""Probe low-corridor recovery from obstructed shelf-clear navigation."""
import sys
from approach import GeneratedApproach
from env_client import make_env

seed=int(sys.argv[1]);env=make_env();s,info=env.reset(seed=seed,options={"object_count":7})
p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info)
for step in range(1000):
    a=p.get_action(s);r=s.get_object_from_name("robot")
    ry=float(s.get(r,"y"))
    if p.stage=="clear_escape_across":
        p.clear_side_x=p.goal_base[0]
        if ry<2.20:
            a=p._action(dy=2.22-ry,vac=0)
        else:
            a=p._action(dx=p.clear_side_x-float(s.get(r,"x")),vac=0)
    if p.stage=="retract" and float(s.get(r,"arm_joint"))<.36:
        if float(s.get(r,"x"))>1.50: a=p._action(dx=-.05,vac=1)
        elif float(s.get(r,"y"))>1.40: a=p._action(dy=-.05,vac=1)
    s,_,term,trunc,_=env.step(a)
    if step%100==99:
        b=s.get_object_from_name(p.target_name)
        print(step+1,p.stage,p.target_name,round(float(s.get(r,"x")),2),round(float(s.get(r,"y")),2),
              "arm",round(float(s.get(r,"arm_joint")),2),"b",round(float(s.get(b,"x")),2),round(float(s.get(b,"y")),2))
    if term or trunc: print("END",step+1,term,trunc);break
env.close()
