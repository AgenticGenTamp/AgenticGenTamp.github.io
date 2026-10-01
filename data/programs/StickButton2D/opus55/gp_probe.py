import math,numpy as np,sys
from env_client import make_env
from exprot import rd, wrap
env=make_env()
SEED=3
def st(obs):
    r,s,_=rd(obs); return r,s
def act(dx=0,dy=0,dth=0,da=0,vac=0.0):
    global obs
    obs,*_=env.step(np.array([dx,dy,dth,da,vac],dtype=np.float32)); return st(obs)
def goto(x,y,th,arm,vac=0.0,n=400):
    for _ in range(n):
        r,s=st(obs)
        ex,ey=x-r['x'],y-r['y']; et=wrap(th-r['theta']); ea=arm-r['arm_joint']
        if abs(ex)<1e-4 and abs(ey)<1e-4 and abs(et)<1e-4 and abs(ea)<1e-4: return True
        act(np.clip(ex,-.05,.05),np.clip(ey,-.05,.05),np.clip(et,-.19,.19),np.clip(ea,-.1,.1),vac)
    return False
def rel(r,s):
    c,sn=math.cos(r['theta']),math.sin(r['theta']); dx,dy=s[0]-r['x'],s[1]-r['y']
    return (c*dx+sn*dy, -sn*dx+c*dy, wrap(s[2]-r['theta']))
def trial(side,d,h=0.0,stopgap=None):
    global obs
    obs,_=env.reset(seed=SEED)
    r,s=st(obs); sx,sy=s[0],s[1]; W=0.05
    if side=='bot': tx,ty,th=sx+W/2+ (0 if h is None else h), sy, math.pi/2+d
    elif side=='left': tx,ty,th=sx, sy+h, d
    else: tx,ty,th=sx+W, sy+h, math.pi+d
    L=0.2; back=0.04
    bx,by=tx-(L+back)*math.cos(th), ty-(L+back)*math.sin(th)
    ok=goto(r['x'],min(r['y'],sy-0.4),th,0.1) and goto(bx,min(by,sy-0.4),th,0.1) and goto(bx,by,th,0.1) and goto(bx,by,th,L)
    if not ok: return 'nav fail', st(obs)[0]
    ux,uy=math.cos(th),math.sin(th); moved=0; contact=None
    for k in range(60):
        r0,s0=st(obs)
        r1,s1=act(0.002*ux,0.002*uy,0,0,1.0)
        if abs(r1['x']-r0['x'])<1e-6 and abs(r1['y']-r0['y'])<1e-6: contact='rejected@%d'%k; break
        if max(abs(s1[0]-s0[0]),abs(s1[1]-s0[1]))>1e-6: contact='stickmoved@%d'%k; break
        if stopgap is not None and k==stopgap: contact='stopped@%d'%k; break
    r0,s0=st(obs); rel0=rel(r0,s0)
    for _ in range(4): r,s=act(0,-0.03,0,0,1.0)
    for _ in range(3): r,s=act(-0.03 if side!='right' else 0.03,0,0,0,1.0)
    r,s=act(0,0,0.1,0,1.0)
    rel1=rel(r,s); dr=[round(a-b,4) for a,b in zip(rel1,rel0)]
    moved=math.hypot(s[0]-s0[0],s[1]-s0[1])
    return contact, 'stick moved %.3f'%moved, 'rel drift',dr, 'robot',round(r['x'],3),round(r['y'],3)
if __name__=='__main__':
    for side,ds,hs in (('bot',[0,.3,-.3,.6,-.6,1.0,-1.0],[0]),('left',[0,.3,-.3,.6,-.6],[0.1,0.3]),('right',[0,.3,-.3,.6,-.6],[0.1,0.3])):
        for h in hs:
            for d in ds:
                print(side,h,d,trial(side,d,h),flush=True)
