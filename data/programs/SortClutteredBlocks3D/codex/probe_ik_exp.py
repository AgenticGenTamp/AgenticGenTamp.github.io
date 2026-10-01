from env_client import make_env
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def snap(s):
    out = {}
    for name in s.get_object_names():
        o = s.get_object_from_name(name)
        if name == "robot":
            fs = ["pos_base_x", "pos_base_y", "pos_base_rot"] + [f"pos_arm_joint{i}" for i in range(1,8)] + ["pos_gripper"]
        else:
            fs = ["x", "y", "z"]
        vals = []
        for f in fs:
            try: vals.append(float(s.get(o, f)))
            except Exception: vals.append(None)
        out[name] = vals
    return out


def main():
    e=make_env(); s,info=e.reset(seed=0, options={"object_count":4})
    print("info",info); print("initial",snap(s)); print("max",e.max_steps)
    acts=[]
    for idx,val in [(0,.1),(1,.1),(2,.1),(3,.1),(4,.1),(5,.1),(6,.1),(7,.1),(8,.1),(9,.1)]:
        a=np.zeros(11,np.float32); a[idx]=val; a[10]=1
        before=snap(s)["robot"]
        s,r,t,tr,i=e.step(a)
        after=snap(s)["robot"]
        print("step",idx,"r",r,"delta",np.array(after)-np.array(before),"robot",after)
    for grip in [0,1,0,1]:
        a=np.zeros(11,np.float32); a[10]=grip
        before=snap(s)["robot"]
        s,r,t,tr,i=e.step(a); after=snap(s)["robot"]
        print("gripcmd",grip,"r",r,"delta",np.array(after)-np.array(before),"pos",after[-1])
    e.close()

def drive(grip):
    e=make_env(); s,info=e.reset(seed=0, options={"object_count":4})
    names=sorted(n for n in s.get_object_names() if n.startswith("cube"))
    def row():
        ss=snap(s); return [round(ss["robot"][0],3),round(ss["robot"][1],3)] + [[round(x,3) for x in ss[n][:3]] for n in names]
    print("DRIVE",grip,"start",row())
    for k in range(14):
        a=np.zeros(11,np.float32); a[0]=-.1; a[10]=grip
        s,r,t,tr,i=e.step(a)
        print(k, r, row())
    e.close()

def sweep(jidx, sign):
    e=make_env(); s,info=e.reset(seed=0, options={"object_count":4})
    names=sorted(n for n in s.get_object_names() if n.startswith("cube"))
    orig={n:np.array(snap(s)[n][:3]) for n in names}
    # translate base to x~.52, y~0 so the home-pose tool is over pile
    for k in range(8):
        ss=snap(s)["robot"]; a=np.zeros(11,np.float32)
        a[0]=np.clip((.53-ss[0])*.8,-.1,.1); a[1]=np.clip((0-ss[1])*.8,-.1,.1); a[10]=0
        s,r,t,tr,i=e.step(a)
    print("SWEEP",jidx,sign,"base",snap(s)["robot"][:3])
    for k in range(15):
        a=np.zeros(11,np.float32); a[jidx]=sign*.1; a[10]=0
        s,r,t,tr,i=e.step(a); ss=snap(s)
        dif=max(np.linalg.norm(np.array(ss[n][:3])-orig[n]) for n in names)
        if k%2==0 or dif>.005: print(k,"q",round(ss["robot"][jidx],3),"r",r,"move",round(dif,4),[[round(x,3) for x in ss[n][:3]] for n in names])
    e.close()

ORIG=[([0,0,.1564],[np.pi,0,0]),([0,.0054,-.1284],[np.pi/2,0,0]),([0,-.2104,-.0064],[-np.pi/2,0,0]),([0,.0064,-.2104],[np.pi/2,0,0]),([0,-.2084,-.0064],[-np.pi/2,0,0]),([0,0,.1059],[np.pi/2,0,0]),([0,-.1059,0],[-np.pi/2,0,0])]
def fk(q):
    T=np.eye(4)
    for qi,(x,r) in zip(q,ORIG):
        A=np.eye(4); A[:3,:3]=Rotation.from_euler('xyz',r).as_matrix(); A[:3,3]=x; T=T@A
        A=np.eye(4); A[:3,:3]=Rotation.from_rotvec([0,0,qi]).as_matrix(); T=T@A
    return T
def ik(pos):
    q0=np.array([0,-.3490659,np.pi,-2.54818,0,-.872665,np.pi/2])
    def err(q):
        T=fk(q); return np.r_[10*(T[:3,3]-pos), 2*(T[:3,2]-[0,0,-1])]
    res=least_squares(err,q0,bounds=(-np.pi,np.pi),max_nfev=3000)
    print("IK",pos,np.round(res.x,3),"got",np.round(fk(res.x)[:3,3],3),"axis",np.round(fk(res.x)[:3,2],2),res.cost)
    return res.x

def pose_trial(pos):
    target=ik(np.array(pos)); e=make_env(); s,info=e.reset(seed=0, options={"object_count":4})
    names=sorted(n for n in s.get_object_names() if n.startswith("cube")); orig={n:np.array(snap(s)[n][:3]) for n in names}
    # approach table but retain safe x=.58
    for k in range(10):
        rob=snap(s)["robot"]; a=np.zeros(11,np.float32); a[0]=np.clip((.58-rob[0])*.8,-.1,.1); a[1]=np.clip((0-rob[1])*.8,-.1,.1); a[10]=0; s,r,t,tr,i=e.step(a)
    for k in range(160):
        rob=snap(s)["robot"]; q=np.array(rob[3:10]); a=np.zeros(11,np.float32); a[3:10]=np.clip((target-q)*.5,-.1,.1); a[10]=0; s,r,t,tr,i=e.step(a)
        ss=snap(s); dif=max(np.linalg.norm(np.array(ss[n][:3])-orig[n]) for n in names)
        if dif>.002 or k in [60,100,159]: print("POSE",pos,k,"r",r,"move",round(dif,3),"q",np.round(ss["robot"][3:10],2),[[round(x,3) for x in ss[n][:3]] for n in names])
    e.close()

def random_search():
    e=make_env(); s,info=e.reset(seed=7, options={"object_count":4}); rng=np.random.default_rng(123)
    names=sorted(n for n in s.get_object_names() if n.startswith("cube")); orig={n:np.array(snap(s)[n][:3]) for n in names}
    for k in range(10):
        rob=snap(s)["robot"]; a=np.zeros(11,np.float32); a[0]=np.clip((.56-rob[0])*.8,-.1,.1); a[1]=np.clip((0-rob[1])*.8,-.1,.1); a[10]=0; s,r,t,tr,i=e.step(a)
    moved=False
    for seg in range(18):
        # broad shoulder/elbow/wrist targets, with base yaw joint near pile center
        target=np.array([rng.uniform(-.5,.5),rng.uniform(-2.2,2.2),rng.uniform(-3.0,3.0),rng.uniform(-2.5,2.2),rng.uniform(-3,3),rng.uniform(-2.2,2.2),rng.uniform(-3,3)])
        print("TARGET",seg,np.round(target,2))
        for k in range(45):
            rob=snap(s)["robot"]; q=np.array(rob[3:10]); a=np.zeros(11,np.float32); a[3:10]=np.clip((target-q)*.7,-.1,.1); a[10]=0; s,r,t,tr,i=e.step(a)
            ss=snap(s); diffs={n:np.linalg.norm(np.array(ss[n][:3])-orig[n]) for n in names}; dif=max(diffs.values())
            if dif>.003:
                print("CONTACT",seg,k,"r",r,"q",np.round(ss["robot"][3:10],3),"diffs",{n:round(v,3) for n,v in diffs.items()},"xyz",{n:[round(x,3) for x in ss[n][:3]] for n in names})
                moved=True; break
        if moved: break
    e.close()

if __name__=="__main__":
    # main()
    # drive(0); drive(1)
    # for j in [4,6,8]:
    #     for sign in [-1,1]: sweep(j,sign)
    # for p in [[.5,0,.3],[.5,0,.15]]: pose_trial(p)
    random_search()
