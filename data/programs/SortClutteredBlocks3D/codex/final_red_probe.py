from env_client import make_env
import numpy as np

Q=np.array([0.,1.3,np.pi,-1.7,0.,1.,0.])

def val(s,n,fs):
    o=s.get_object_from_name(n); return np.array([float(s.get(o,f)) for f in fs])
def rob(s): return val(s,"robot",["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"])
def xyz(s,n): return val(s,n,["x","y","z"])
def act(s,b=None,q=None,g=0):
    r=rob(s); a=np.zeros(11,np.float32)
    if b is not None: a[:3]=np.clip(.8*(np.asarray(b)-r[:3]),-.1,.1)
    if q is not None: a[3:10]=np.clip(.7*(np.asarray(q)-r[3:10]),-.1,.1)
    a[10]=g; return a
def run(e,s,n,b=None,q=None,g=0):
    last=None
    for _ in range(n): s,*last=e.step(act(s,b,q,g))
    return s,last

def main():
    e=make_env(); s,info=e.reset(seed=0,options={"object_count":4})
    home=rob(s)[3:10].copy(); c="cube1"; red=xyz(s,"bin_red").copy()
    bins={n:xyz(s,n).copy() for n in ["bin_red","bin_green","bin_blue","bin_yellow"]}
    print("START",c,np.round(xyz(s,c),4),"red",np.round(red,4),"home",np.round(home,3))
    # Safe route to the left side, then offset +4.2cm in y for the q1+ rake.
    s,_=run(e,s,45,[.99,.8,np.pi],home)
    s,_=run(e,s,55,[-1,.8,np.pi],home)
    s,_=run(e,s,45,[-1,.8,0],home)
    y=float(xyz(s,c)[1]+.042)
    s,_=run(e,s,45,[-1,y,0],home)
    s,_=run(e,s,130,[-1,y,0],Q)
    initial=xyz(s,c).copy(); all0={n:xyz(s,n).copy() for n in ["cube1","cube2","cube3","cube4"]}; hit=None
    for k in range(50):
        a=act(s,[-.8,y,0],Q); a[0]=min(a[0],.012)
        s,r,t,tr,i=e.step(a)
        if max(np.linalg.norm(xyz(s,n)-v) for n,v in all0.items())>.001:
            hit=rob(s)[:3].copy(); print("LEFT_HIT",k,np.round(hit,4),"cube1",np.round(xyz(s,c),4),"all",{n:np.round(xyz(s,n),4).tolist() for n in all0}); break
    # Tangentially sweep cube1 below yellow-bin y.
    qt=Q.copy(); qt[0]=.8
    for k in range(120):
        s,r,t,tr,i=e.step(act(s,hit,qt)); p=xyz(s,c)
        if k%10==0: print("LEFT_SWEEP",k,"q1",round(rob(s)[3],3),"cube",np.round(p,4),"r",r)
        if p[1] <= -.285: print("BELOW",k,"q1",round(rob(s)[3],4),"cube",np.round(p,4),"bins",{n:np.round(xyz(s,n)-v,4).tolist() for n,v in bins.items()}); break
    low=xyz(s,c).copy()
    # Disengage left, fold completely, and take the outside route to right.
    s,_=run(e,s,20,[-1.10,rob(s)[1],0],qt)
    s,_=run(e,s,100,[-1.10,-.8,0],home)
    s,_=run(e,s,55,[1.0,-.8,0],home)
    s,_=run(e,s,45,[1.0,-.8,np.pi],home)
    yr=float(xyz(s,c)[1])
    s,_=run(e,s,45,[1.0,yr,np.pi],home)
    s,_=run(e,s,130,[1.0,yr,np.pi],Q)
    print("RIGHT_READY","robot",np.round(rob(s),3),"cube",np.round(xyz(s,c),4),"bins",{n:np.round(xyz(s,n)-v,4).tolist() for n,v in bins.items()})
    # Approach from +x, checking both cube and bin motion.
    before=xyz(s,c).copy(); contact=None
    for k in range(60):
        a=act(s,[.65,yr,np.pi],Q); a[0]=max(a[0],-.012)
        s,r,t,tr,i=e.step(a); p=xyz(s,c)
        moved={n:float(np.linalg.norm(xyz(s,n)-v)) for n,v in bins.items()}
        if max(moved.values())>.001: print("BIN_CONTACT",k,{n:round(v,4) for n,v in moved.items()})
        if np.linalg.norm(p-before)>.001 and contact is None:
            contact=rob(s)[:3].copy(); print("RIGHT_HIT",k,np.round(contact,4),"cube",np.round(p,4),"bins",{n:round(v,4) for n,v in moved.items()})
        if contact is not None and p[0] <= red[0]+.005:
            print("X_REACHED",k,"cube",np.round(p,4),"dist",round(float(np.linalg.norm(p-red)),4),"r",r); break
    # Maintain inward x pressure and translate the holonomic base laterally,
    # using the rake as a straight guide toward red-bin y.
    press=rob(s)[:3].copy(); desired_base_y=press[1] + (red[1]-xyz(s,c)[1])
    for k in range(50):
        b=press.copy(); b[1]=desired_base_y
        a=act(s,b,Q); a[1]=np.clip(a[1],-.018,.018); a[0]=-.006
        s,r,t,tr,i=e.step(a); p=xyz(s,c); d=float(np.linalg.norm(p-red))
        if k%5==0: print("Y_GUIDE",k,"cube",np.round(p,4),"dist",round(d,4),"r",r)
        if d<.05: print("RED_REACHED",k,"cube",np.round(p,4),"dist",round(d,4),"r",r); break
    p=xyz(s,c); print("FINAL","steps_cube",np.round(p,4),"red_dist",round(float(np.linalg.norm(p-red)),4),"robot",np.round(rob(s),3),"bins",{n:np.round(xyz(s,n)-v,4).tolist() for n,v in bins.items()},"last_r",r)
    e.close()

if __name__=="__main__": main()
