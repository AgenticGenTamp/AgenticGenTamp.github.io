from lib import *
e=E(3)  # robot at 0.548,0.454, hook far right
for d,name in [((0,-1),'down'),((-1,0),'left'),((0,1),'up')]:
    for step in [0.05,0.01,0.002]:
        while True:
            p=e.obs[:2].copy(); e.st([d[0]*step,d[1]*step,0,0,0])
            if np.allclose(p,e.obs[:2]): break
    print(name,e.obs[:3])
