import sys, numpy as np
from env_client import make_env
from ctrl import Bot
np.set_printoptions(precision=4,suppress=True)
G=float(sys.argv[1])
b=Bot(make_env(),0); SL=32; can=b.obs[SL:SL+3].copy()
b.grip(G,5)
# drive base to 0.45 m in -x from can, aligned in y
tgt=np.array([can[0]-0.45, can[1]])
for _ in range(60):
    d=tgt-b.obs[93:95]
    if np.linalg.norm(d)<0.01: break
    a=np.zeros(11); a[0:2]=np.clip(d/0.87,-0.1,0.1); a[2]=np.clip(-b.obs[95],-0.1,0.1); a[10]=G
    b.step(a)
print("base now",b.obs[93:96],"can",b.obs[SL:SL+3])
def probe(xy,tag):
    e,err=b.servo(np.array([xy[0],xy[1],0.72]),G,200,tol=0.004)
    print(tag,"above EE",e,"err %.4f"%err)
    blk=None
    for z in np.arange(0.71,0.42,-0.01):
        e,err=b.servo(np.array([xy[0],xy[1],z]),G,25,tol=0.003)
        if err>0.02: blk=e[2]; break
    print("%s BLOCK_FKZ=%s cmd_z=%.3f err=%.4f can=%s"%(tag,None if blk is None else round(blk,4),z,err,b.obs[SL:SL+3]))
    return blk
tb=probe(np.array([can[0],can[1]+0.20]),"TABLE")
b.servo(np.array([can[0],can[1]+0.20,0.72]),G,120,tol=0.01)
cn=probe(can[:2],"CANTOP")
print("RES G=%.1f table=%s cantop=%s diff=%s canmoved=%.4f steps=%d"%(G,tb,cn,None if (tb is None or cn is None) else round(cn-tb,4),np.linalg.norm(b.obs[SL:SL+3]-can),b.n))
b.env.close()
