import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *; from h import _quat_yaw, _wrap
seed=int(sys.argv[1]); yaw=float(sys.argv[2]); pz=0.488
h=H(seed); h.step(60); dp=h.drink()
h.setbase([-0.30, dp[1], 0]); h.move([dp[0],dp[1],0.60], yaw, 0.03); dp=h.drink()
h.move([dp[0],dp[1],pz], yaw, 0.01); h.close(25); h.rep("closed")
m,d=h.m,h.d; f6=np.zeros(6)
def grip_forces():
    tot=np.zeros(3); n=0
    for i in range(d.ncon):
        c=d.contact[i]; b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
        if h.bid in (b1,b2):
            other=b2 if b1==h.bid else b1; nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,other) or ""
            if 'kitchen' in nm: continue
            mujoco.mj_contactForce(m,d,i,f6); fr=c.frame.reshape(3,3)
            w=fr.T@f6[:3]; tot+= (w if b1==h.bid else -w)*(-1); n+=1
    return n,tot
print("at grip", grip_forces())
p0,_=h.pinch()
for k in range(30):
    h.move([dp[0],dp[1],pz+0.002*(k+1)], yaw, 0.01, settle=3)
    if k%3==0: n,t=grip_forces(); print("cmd dz",0.002*(k+1),"pinch z",round(h.pinch()[0][2],3),"drink z",round(h.drink()[2],3),"ncon",n,"force on drink",np.round(t,2))
