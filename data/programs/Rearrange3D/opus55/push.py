from probe_lib import *
import sys
seed=int(sys.argv[1]); zoff=float(sys.argv[2]); ds=float(sys.argv[3])
pr=P(seed)
for i in range(60): pr.step(np.zeros(11))
o=pr.obs; D=o[16:18].copy(); B=o[0:2].copy(); tz=0.46
T=D+0.11*(B-D)/np.linalg.norm(B-D)
pr.grip=1
for it in range(6):
    B=pr.obs[0:2].copy(); v=T-B; L=np.linalg.norm(v)
    print(it,'bowl',B.round(3),'dist to T',round(L,3), 'bowl z',pr.obs[2].round(3),'qw',pr.obs[3].round(3))
    if L<0.01: break
    u=v/L; start=B-u*0.09; end=B+u*min(L,0.12)-u*0.07
    psi=np.arctan2(u[1],u[0])+np.pi/2
    pr.goto(pr.ik([start[0],start[1],0.56],psi))
    pr.goto(pr.ik([start[0],start[1],tz+zoff],psi))
    p,why,qe=pr.line([start[0],start[1],tz+zoff],[end[0],end[1],tz+zoff],psi,ds=ds,qthr=0.1)
    pr.goto(pr.ik([end[0],end[1],0.56],psi))
print('final bowl',pr.obs[0:3].round(3),'d drink',np.linalg.norm(pr.obs[0:2]-pr.obs[16:18]).round(3),'steps',pr.t if hasattr(pr,'t') else '')
