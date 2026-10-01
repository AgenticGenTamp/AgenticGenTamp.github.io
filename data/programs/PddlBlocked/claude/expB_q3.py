import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
base=np.array([3.72,0.10,0.0])

def grasp_blocker(env):
    obs,s,g,b,d3,yaw=setup(env,1)
    R=fk.grasp_R(yaw); tot=0
    obs,rej,n=step_to(env,obs,base,robot(obs)[3:10]); tot+=n
    q_pre,_=fk.ik(b-d3*0.20,R,base,robot(obs)[3:10],seeds=6)
    obs,r1,n=step_to(env,obs,base,q_pre); tot+=n
    q_g,_=fk.ik(b-d3*0.03,R,base,q_pre,seeds=6)
    obs,r2,n=step_to(env,obs,base,q_g); tot+=n
    obs=grip(env,obs,-1.0); tot+=1
    return obs,s,g,b,d3,yaw,R,tot,(r1,r2)

def report(tag,obs,before,tot):
    ga=robot(obs)[11]
    print(f"{tag}: grasp_active={ga} steps={tot}")
    for k in before:
        d=np.linalg.norm(blockpos(obs,k)-before[k])
        if d>1e-4: print(f"   moved {k} by {d:.4f} -> {np.round(blockpos(obs,k),3).tolist()}")

for variant in ["retract025_lat+","retract025_lat-","retract035_only","retract015_lat-"]:
    env=make_env()
    obs,s,g,b,d3,yaw,R,tot,rr=grasp_blocker(env)
    print("="*10,variant,"grasp of blocker: active=",robot(obs)[11],"steps",tot,"rej",rr)
    if robot(obs)[11]<0.5:
        env.close(); continue
    perp=np.array([-d3[1],d3[0],0.0])
    before={k:blockpos(obs,k) for k in names(obs) if k!="robot"}
    q=robot(obs)[3:10]
    if variant=="retract035_only":
        wp=[b-d3*0.35]
    elif variant=="retract015_lat-":
        wp=[b-d3*0.15, b-d3*0.15-perp*0.3]
    else:
        sgn=1.0 if variant.endswith("+") else -1.0
        wp=[b-d3*0.25, b-d3*0.25+sgn*perp*0.3]
    ok=True
    for w in wp:
        qn,e=fk.ik(w,R,base,q,seeds=8)
        if e>0.02: print("   IK fail",np.round(w,3),e); ok=False; break
        obs,rej,n=step_to(env,obs,base,qn,grip=-1.0,maxsteps=40); tot+=n; q=qn
        p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
        print(f"   wp {np.round(w,3).tolist()} rej={rej} steps={n} tool={np.round(p,3).tolist()} ga={robot(obs)[11]}")
        if rej: ok=False; break
    obs=grip(env,obs,1.0); tot+=1
    print(f"   after open: grasp_active={robot(obs)[11]} (REFUSED={robot(obs)[11]>0.5}) opening={robot(obs)[10]:.3f} total_steps={tot}")
    # extra open steps
    for i in range(3):
        obs=grip(env,obs,1.0); tot+=1
    print(f"   after 3 more opens: grasp_active={robot(obs)[11]} blocker={np.round(blockpos(obs,'blocker'),3).tolist()} total={tot}")
    env.close(); sys.stdout.flush()
