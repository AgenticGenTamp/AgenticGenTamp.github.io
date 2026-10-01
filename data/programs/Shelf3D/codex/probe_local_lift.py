"""Local q7/grip/base-offset refinement around the verified q4=-1 lift."""
import sys
import numpy as np
from env_client import make_env
from probe_alt_branches import drive, qpos, cube_pos, get
from solve_alt_fk import fk

BASE_Q=np.array([0.0,2.24,2.945,-1.0,-.982,.2,1.571])
CASES=[
    (1.30,.60,0.00,0.00),(1.30,1.00,0.00,0.00),
    (1.57,.60,0.00,0.00),(1.57,1.00,0.00,0.00),
    (1.85,.60,0.00,0.00),(1.85,1.00,0.00,0.00),
    (1.57,.75,-.03,0.00),(1.57,.75,.03,0.00),
    (1.57,.75,0.00,-.03),(1.57,.75,0.00,.03),
]
if "--best" in sys.argv:
    CASES = CASES[:1]
seed = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--seed=")), 0)
dx_override = next((float(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--dx=")), None)
if dx_override is not None:
    q7, grip, _, dy = CASES[0]
    CASES = [(q7, grip, dx_override, dy)]
grip_override = next((float(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--grip=")), None)
if grip_override is not None:
    q7, _, dx, dy = CASES[0]
    CASES = [(q7, grip_override, dx, dy)]

for q7,grip,dx,dy in CASES:
    env=make_env();state,_=env.reset(seed=seed,options={"object_count":1});cube="cube1"
    c0=cube_pos(state,cube); b0=np.array([get(state,"robot",f) for f in ("pos_base_x","pos_base_y")])
    target=BASE_Q.copy();target[6]=q7
    state=drive(env,state,base_target=b0+[-.15,0],steps=40)
    state=drive(env,state,q_target=target,base_target=b0+[-.15,0])
    point=fk(qpos(state))[:3,3]
    bt=c0[:2]-np.array([-point[0],point[1]])+[dx,dy]
    state=drive(env,state,q_target=target,base_target=bt)
    ca=cube_pos(state,cube)
    contact_step = None
    if "--scan" in sys.argv:
        scan0 = ca.copy()
        for scan_step in range(35):
            a=np.zeros(11,np.float32);cur=qpos(state)
            a[3:10]=np.clip(.35*(target-cur),-.1,.1)
            a[0]=.01; a[10]=0.0
            state,*_=env.step(a)
            if np.linalg.norm(cube_pos(state,cube)-scan0)>.002:
                contact_step=scan_step;break
        # Treat the detected contact pose as the base hold target.
        bt=np.array([get(state,"robot",f) for f in ("pos_base_x","pos_base_y")])
        ca=cube_pos(state,cube)
    state=drive(env,state,q_target=target,base_target=bt,grip=grip,steps=18)
    cc=cube_pos(state,cube)
    lift=target.copy();lift[1]=1.45
    state=drive(env,state,q_target=lift,base_target=bt,grip=grip,steps=100)
    cl=cube_pos(state,cube)
    # A true capture should approximately follow a short base translation.
    base_now=np.array([get(state,"robot",f) for f in ("pos_base_x","pos_base_y")])
    state=drive(env,state,q_target=lift,base_target=base_now+[.12,0],grip=grip,steps=35)
    carry=cube_pos(state,cube)
    print("case",q7,grip,dx,dy,"actual",np.round(qpos(state),2).tolist(),
          "z",round(ca[2],4),round(cc[2],4),round(cl[2],4),
          "dxyz",np.round(cl-c0,4).tolist(),"carry",np.round(carry,4).tolist(),
          "contact_step",contact_step)
    env.close()
