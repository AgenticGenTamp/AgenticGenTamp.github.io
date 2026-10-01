import numpy as np
from env_client import make_env
from probe_fk_ctl import servo, grip_hold, ee_world
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
C0=o[:80].reshape(5,16)[:,:3].copy(); W0=o[147:150].copy(); tot=0
def rep(t,o,e,u,extra=""):
    global tot; tot+=u; c=o[:80].reshape(5,16)[:,:3]
    print(f"{t:12s} ee={np.round(ee_world(o),4)} err={e:.4f} u={u} tot={tot} dc={np.round(np.linalg.norm(c-C0,axis=1),3)} dw={np.linalg.norm(o[147:150]-W0):.3f} {extra}")
t=C0[3]
o,e,u=servo(env,o,[t[0],t[1],0.58],grip=0.0); rep("above3",o,e,u)
o,e,u=servo(env,o,[t[0],t[1],0.4698],grip=0.0); rep("at3",o,e,u,f"cube3={np.round(o[:80].reshape(5,16)[3,:3],4)}")
o=grip_hold(env,o,1.0,15); tot+=15
o,e,u=servo(env,o,[t[0],t[1],0.65],grip=1.0); rep("lift",o,e,u)
c=o[:80].reshape(5,16)[3,:3]; print("   GRASP resid cube-ee:",np.round(c-ee_world(o),4))
GOAL=np.array([0.84,-0.30,0.4698])
o,e,u=servo(env,o,[GOAL[0],GOAL[1],0.65],grip=1.0); rep("carry",o,e,u)
c=o[:80].reshape(5,16)[3,:3]; print("   GRASP resid cube-ee:",np.round(c-ee_world(o),4))
o,e,u=servo(env,o,GOAL,grip=1.0); rep("place",o,e,u)
c=o[:80].reshape(5,16)[3,:3]; print("   GRASP resid cube-ee:",np.round(c-ee_world(o),4))
o=grip_hold(env,o,0.0,10); tot+=10
o,e,u=servo(env,o,[GOAL[0],GOAL[1],0.62],grip=0.0); rep("retreat",o,e,u)
print(">>> PLACE: goal",GOAL[:2],"cube3 final",np.round(o[:80].reshape(5,16)[3,:3],4))
# wiper: touch at a commanded point, push +x
o,e,u=servo(env,o,[0.83,-0.40,0.58],grip=1.0); rep("wip_above",o,e,u)
o,e,u=servo(env,o,[0.83,-0.40,0.468],grip=1.0); rep("wip_touch",o,e,u)
w1=o[147:150].copy()
o,e,u=servo(env,o,[0.93,-0.40,0.468],grip=1.0,steps=120); rep("wip_push+x",o,e,u)
print(">>> WIPER:",np.round(W0,4),"->",np.round(o[147:150],4),"delta",np.round(o[147:150]-W0,4))
# push cube 0 in -y by 8cm
o,e,u=servo(env,o,[C0[0][0],C0[0][1]+0.09,0.60],grip=1.0); rep("c0_above",o,e,u)
o,e,u=servo(env,o,[C0[0][0],C0[0][1]+0.09,0.4698],grip=1.0); rep("c0_down",o,e,u)
o,e,u=servo(env,o,[C0[0][0],C0[0][1]-0.05,0.4698],grip=1.0,steps=120); rep("c0_push-y",o,e,u)
print(">>> CUBE0:",np.round(C0[0],4),"->",np.round(o[:80].reshape(5,16)[0,:3],4))
print("base end",np.round(o[125:128],4))
env.close()
