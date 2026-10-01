from lib import *
import sys
ok=0
for seed in range(int(sys.argv[1]),int(sys.argv[2])):
    e=E(seed); h0=e.obs[9:12].copy()
    e.grasp_hook(1.2)
    o=e.obs
    e.st([0,-0.05,0,0,1]); moved=not np.allclose(e.obs[9:12],o[9:12])
    e.st([0,0.05,0,0,1]);moved2=not np.allclose(e.obs[9:12],o[9:12])
    print(seed,'t',e.t,'off',e.off,'dth',round(e.dth,3),'hookmoved',moved or moved2, 'hook0',h0)
