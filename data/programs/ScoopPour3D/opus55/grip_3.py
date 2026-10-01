import sys
from grip_util import *
from multiprocessing import Pool
def run(cfg):
    ki,sv=cfg; g=G(0); g.ki=ki; g.settle_v=sv
    c=g.P('cube_17'); Rw=kin.rotz(gyaw(g,'cube_17'))@RD; bt=np.array([-0.15,c[1],0])
    g.ctrl(c+[0,0,0.56-c[2]],Rw,bt,tol=0.01,maxsteps=120)
    out=[]
    for dz in [0.03,0.01]:
        g.trace=[]; g.ctrl(c+[0,0,dz],Rw,bt,tol=0.0,maxsteps=40)
        out.append(np.round(np.array(g.trace)*1000,1))
    return cfg,out
with Pool(4) as p:
    for cfg,out in p.imap_unordered(run,[(0.5,1.0),(0.5,0.003),(0.3,0.003),(0.2,1.0)]):
        print(cfg); [print(' ',o.tolist()) for o in out]
