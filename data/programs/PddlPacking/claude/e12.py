import numpy as np, fk, ctrl, lib
from env_client import make_env
env=make_env(); R=lib.Runner(env,seed=0)
print("start tool",np.round(R.tool(),4))
print("base",R.move_base((-0.75,-0.2,0.0)), np.round(R.rob()[:3],3))
ang=np.pi/2
ok=R.cart(np.array([0.55,0.2,0.80]),ang)
print("above free spot",ok,np.round(R.tool(),4))
# descend in 0.01
z=0.80
while z>0.55:
    z-=0.01
    ok=R.cart(np.array([0.55,0.2,z]),ang,maxit=6,step_len=0.02)
    if not ok:
        print("stopped at z(fk)=",round(z+0.01,3),"tool",np.round(R.tool(),4)); break
else:
    print("descended to 0.55!?")
env.close()
