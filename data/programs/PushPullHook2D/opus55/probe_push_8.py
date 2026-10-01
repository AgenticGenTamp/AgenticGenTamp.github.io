from lib import *
import sys
seed=int(sys.argv[1])
e=E(seed); o=e.obs
print('robot',o[:9],'btn',o[20:22],'tgt',o[29:31],'hook',o[9:12])
bx,by=o[20:22]
e.goto(bx, min(by-0.6,1.0), np.pi/2, arm=o[5])
print('at',e.obs[:9],'btn',e.obs[20:22])
for i in range(150):
    p=e.obs[20:22].copy(); pr=e.obs[:2].copy(); te=e.st([0,0.005,0,0,0])
    if not np.allclose(p,e.obs[20:22]): print(i,'robot',e.obs[:2],'arm',e.obs[4],'btn d',e.obs[20:22]-p,'rel',e.obs[21]-e.obs[1]); break
    if np.allclose(pr,e.obs[:2]): print('blocked',i,e.obs[:5],'btn',e.obs[20:22]); break
