import numpy as np, sys
from env_client import make_env
from probe_fk_world import servo_world, ee_world
np.set_printoptions(precision=4,suppress=True,linewidth=250)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
C=o[:80].reshape(5,16)[:,:3].copy(); W0=o[147:150].copy()
tot=0
def rep(tag,o,e,u):
    global tot; tot+=u
    c=o[:80].reshape(5,16)[:,:3]
    print(f"{tag:16s} ee={np.round(ee_world(o),4)} perr={e:.4f} u={u} tot={tot} dcube={np.round(np.linalg.norm(c-C,axis=1),3)} dwip={np.linalg.norm(o[147:150]-W0):.3f}")
# --- accuracy check at free-space points
for p in [[0.75,-0.15,0.62],[0.85,-0.35,0.60],[0.70,-0.05,0.65],[0.90,-0.20,0.55]]:
    o,e,u=servo_world(env,o,p,steps=150,grip=0.0); rep(f"free{p[0]},{p[1]}",o,e,u)
    print("   cmd",np.round(p,3),"err_vec",np.round(ee_world(o)-np.array(p),4))
# --- wiper contact: press down onto wiper center then push -x
o,e,u=servo_world(env,o,[0.885,-0.385,0.58],steps=150,grip=1.0); rep("wiper_above",o,e,u)
o,e,u=servo_world(env,o,[0.885,-0.385,0.468],steps=120,grip=1.0); rep("wiper_touch",o,e,u)
o,e,u=servo_world(env,o,[0.80,-0.385,0.468],steps=150,grip=1.0); rep("wiper_push-x",o,e,u)
print("wiper",np.round(W0,4),"->",np.round(o[147:150],4))
env.close()
