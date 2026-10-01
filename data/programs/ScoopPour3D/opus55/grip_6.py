from grip_util import *
g=G(0); n='cube_17'
orig=g.step; H=[]
def st(db=(0,0,0),dq=None):
    r=orig(db,dq); rb=g.rob()
    H.append((g.g, round(g.obs.get(rb,'vel_gripper'),3), (g.tool()*1000).round(1).tolist(), (g.P(n)*1000).round(1).tolist()))
    return r
g.step=st
c0=g.P(n); Rw=kin.rotz(gyaw(g,n))@RD; bt=np.array([-0.15,c0[1],0])
print('yaw',gyaw(g,n))
g.ctrl(c0+[0,0,0.56-c0[2]],Rw,bt,tol=0.01,maxsteps=120)
g.ctrl(c0+[0,0,0.03],Rw,bt,tol=0.004,maxsteps=60)
H.clear()
g.ctrl(c0+[0,0,0.0],Rw,bt,tol=0.002,maxsteps=60)
for h in H: print('desc',h)
H.clear(); g.g=1.0; g.hold(None,Rw,bt,12)
for h in H: print('close',h)
H.clear(); g.ctrl(c0+[0,0,0.56-c0[2]],Rw,bt,tol=0.02,maxsteps=60)
for h in H[:20]: print('lift',h)
