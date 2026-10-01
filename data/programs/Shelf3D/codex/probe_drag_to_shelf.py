"""Repeat the verified ground-drag contact toward the cupboard."""
import numpy as np
from env_client import make_env
from probe_alt_branches import drive, qpos, cube_pos, get
from solve_alt_fk import fk

env=make_env(); state,_=env.reset(seed=0,options={"object_count":1}); cube="cube1"
low=np.array([0,2.24,3.14,-1,0,-.3,-2.25])
lift=np.array([0,.70,3.139,-1.90,.005,-.60,-2.25])
for cycle in range(14):
    cp=cube_pos(state,cube)
    base=np.array([get(state,"robot",f) for f in ("pos_base_x","pos_base_y")])
    state=drive(env,state,q_target=lift,base_target=base+[-.12,0],grip=.6,steps=55)
    state=drive(env,state,q_target=low,grip=0,steps=90)
    point=fk(qpos(state))[:3,3]
    target=cp[:2]-np.array([-point[0],point[1]])+[-.02,0]
    state=drive(env,state,q_target=low,base_target=target,grip=0,steps=55)
    state=drive(env,state,q_target=lift,grip=.6,steps=65)
    current=cube_pos(state,cube)
    print(cycle,np.round(current,3))
    if current[2] > .05:
        # Nudge forward while airborne so it clears the cupboard's front lip.
        for _ in range(40):
            action=np.zeros(11,np.float32)
            if cube_pos(state,cube)[0] < 1.45:
                action[0]=.03; action[10]=.6
            state,reward,term,trunc,_=env.step(action)
            obj=state.get_object_from_name(cube)
            print("impulse",_,np.round(cube_pos(state,cube),3),
                  round(float(state.get(obj,"vx")),3),
                  round(float(state.get(obj,"vz")),3))
        # Freeze/open and let each climb settle before the next.
        for settle in range(15):
            state,reward,term,trunc,_=env.step(np.zeros(11,np.float32))
        print("settled",round(reward,3),np.round(cube_pos(state,cube),3),term)
        if term: break
env.close()
