import sys, numpy as np
from env_client import make_env
from ctrl import Bot, Rd
np.set_printoptions(precision=4,suppress=True)
G=float(sys.argv[1]); MODE=sys.argv[2]
b=Bot(make_env(),0); SL=32; can=b.obs[SL:SL+3].copy()
xy = can[:2] if MODE=='can' else np.array([can[0],can[1]+0.17])
b.grip(G,8)
e,err=b.servo(np.array([xy[0],xy[1],0.70]),G,200,tol=0.004)
print("mode",MODE,"G",G,"above EE",e,"err %.4f"%err,"steps",b.n)
blocked=None
for z in np.arange(0.69,0.42,-0.01):
    e,err=b.servo(np.array([xy[0],xy[1],z]),G,25,tol=0.003)
    if err>0.02:
        blocked=(z,e[2]); print("BLOCKED cmd_z=%.3f fkz=%.4f err=%.4f can=%s"%(z,e[2],err,b.obs[SL:SL+3])); break
if blocked is None: print("no block final",e,err)
print("RES G=%.1f MODE=%s BLOCK_FKZ=%s canmoved=%.4f steps=%d"%(G,MODE,None if blocked is None else round(blocked[1],4),np.linalg.norm(b.obs[SL:SL+3]-can),b.n))
b.env.close()
