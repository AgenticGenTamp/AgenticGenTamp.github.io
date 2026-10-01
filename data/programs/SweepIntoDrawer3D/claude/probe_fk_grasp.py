import numpy as np, sys
from env_client import make_env
from probe_fk_world import move_world, ee_world
np.set_printoptions(precision=4,suppress=True,linewidth=250)
ZG=float(sys.argv[1]) if len(sys.argv)>1 else 0.462
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
C=o[:80].reshape(5,16)[:,:3].copy(); tgt=C[3].copy()
def step_n(o,n,g):
    for _ in range(n):
        a=np.zeros(11); a[10]=g; o,_,_,_,_=env.step(a); o=np.asarray(o,float)
    return o
def show(tag,o,je,u):
    c=o[:80].reshape(5,16)[:,:3]
    print(f"{tag} ee={np.round(ee_world(o),4)} cube3={np.round(c[3],4)} jerr={je:.4f} u={u} grip={o[135]:.2f}")
o,je,u,_=move_world(env,o,[tgt[0],tgt[1],0.56],steps=250,grip=0.0); show("above",o,je,u)
o,je,u,_=move_world(env,o,[tgt[0],tgt[1],ZG],steps=200,grip=0.0); show("at_cube",o,je,u)
o=step_n(o,15,1.0); show("closed",o,0,15)
o,je,u,_=move_world(env,o,[tgt[0],tgt[1],0.62],steps=200,grip=1.0); show("lift",o,je,u)
for p in [[0.80,-0.10,0.62],[0.80,-0.25,0.62],[0.72,-0.25,0.70],[0.86,-0.30,0.55]]:
    o,je,u,_=move_world(env,o,p,steps=200,grip=1.0); show(f"mv{p}",o,je,u)
    print("   qerr_per_joint n/a; q=",np.round(o[128:135],3))
env.close()
