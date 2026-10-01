from far_util import mk
from far_recipe import far_grasp
from env_client import make_env
env=make_env()
for s in [1,35,44,53]:
    c=mk(env,s); hx=c.g('x',c.H)
    h=far_grasp(c)
    print("seed",s,"hx",round(hx,4),"HELD",h,"steps",c.t,flush=True)
env.close(); print("DONE")
