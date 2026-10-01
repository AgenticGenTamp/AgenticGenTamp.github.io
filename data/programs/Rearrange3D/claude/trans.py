import numpy as np
from env_client import make_env
from ctrl import Bot
np.set_printoptions(precision=5,suppress=True)
b=Bot(make_env(),0); SL=32; can=b.obs[SL:SL+3].copy(); Z=0.593
b.grip(1.0,5)
b.servo(np.array([can[0],can[1]+0.17,0.70]),1.0,200,tol=0.004)
b.servo(np.array([can[0],can[1]+0.17,Z]),1.0,140,tol=0.004)
e,err=b.servo(np.array([can[0],-0.34,Z]),1.0,60,tol=0.002)
print("pre-close EE",e,"can",b.obs[SL:SL+3])
p=b.obs[SL:SL+3].copy(); a=np.zeros(11); a[10]=0.0; tr=[]
for i in range(25): b.step(a); tr.append(round(float(np.linalg.norm(b.obs[SL:SL+3]-p)),5))
print("CLOSE(1->0) cumulative can displacement by step:", tr)
print("obs103", b.obs[103], "| can vel obs[39:42]", b.obs[39:42])
p=b.obs[SL:SL+3].copy(); a[10]=1.0; tr=[]
for i in range(25): b.step(a); tr.append(round(float(np.linalg.norm(b.obs[SL:SL+3]-p)),5))
print("OPEN(0->1) cumulative can displacement by step:", tr)
print("steps",b.n)
b.env.close()
