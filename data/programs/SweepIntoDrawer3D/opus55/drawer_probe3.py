from drawer_util import *
import sys
ys=[float(v) for v in sys.argv[1:]]
for y in ys:
    r=R()
    bd=r.base.copy(); bd[0]=1.6; bd[1]=y
    r.goto(r.q,40,base_d=bd)
    # face at z=0.30
    q,e=r.ik([1.1,y,0.30],RA); r.goto(q,80)
    x=1.1; face=None
    while x>0.6:
        x-=0.02
        q,e=r.ik([x,y,0.30],RA); err=r.goto(q,10)
        if err>0.02: face=r.tool()[0,3]; break
    q,e=r.ik([1.1,y,0.30],RA); r.goto(q,30)
    # vertical sweep down at x=0.975
    q,e=r.ik([1.1,y,0.44],RA); r.goto(q,40)
    q,e=r.ik([0.975,y,0.44],RA); err0=r.goto(q,40)
    z=0.44; hits=[]
    while z>0.04:
        z-=0.01
        q,e=r.ik([0.975,y,z],RA); err=r.goto(q,8)
        if err>0.02:
            hits.append(round(r.tool()[2,3],3))
            # go around: back out, jump 6cm down
            q,e=r.ik([1.1,y,z],RA); r.goto(q,25); z-=0.06
            q,e=r.ik([1.1,y,z],RA); r.goto(q,25)
    print('y %.2f face_x %s err0 %.3f handle_top_hits %s dr %s steps %d rew %s'%(y,face,err0,hits,r.obs[103:109],r.n,set(r.rews)),flush=True)
