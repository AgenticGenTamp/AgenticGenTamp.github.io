from env_client import make_env
import numpy as np

Q = np.array([0.0, 1.3, np.pi, -1.7, 0.0, 1.0, 0.0])

def get(s, name, fs):
    o=s.get_object_from_name(name)
    return np.array([float(s.get(o,f)) for f in fs])

def robot(s):
    return get(s,"robot",["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"])

def cubes(s):
    return {n:get(s,n,["x","y","z"]) for n in s.get_object_names() if n.startswith("cube")}

def action_to(s, base=None, q=None, grip=0):
    a=np.zeros(11,np.float32); r=robot(s)
    if base is not None:
        a[:3]=np.clip((np.asarray(base)-r[:3])*.8,-.1,.1)
    if q is not None:
        a[3:10]=np.clip((np.asarray(q)-r[3:10])*.7,-.1,.1)
    a[10]=grip
    return a

def settle_to(e,s,b,q,n=100,grip=0):
    for _ in range(n): s,_,_,_,_=e.step(action_to(s,b,q,grip))
    return s

def lateral_trial(sign, grip=0):
    e=make_env(); s,info=e.reset(seed=0,options={"object_count":4}); initial=cubes(s)
    initial_bins={n:get(s,n,["x","y","z"]) for n in s.get_object_names() if n.startswith("bin_")}
    # Go around table, reaching left side without crossing any fixtures.
    # Keep the arm folded while driving the long route; the rake pose itself
    # otherwise catches the table during transit.
    s=settle_to(e,s,[.99,.80,3.14159],None,45,grip)
    print("ROUTE1",np.round(robot(s)[:3],3))
    s=settle_to(e,s,[-1.0,.80,3.14159],None,55,grip)
    print("ROUTE2",np.round(robot(s)[:3],3))
    s=settle_to(e,s,[-1.0,.80,0],None,45,grip)
    print("ROUTE3",np.round(robot(s)[:3],3))
    # Positive q1 sweeps the edge toward -y, negative toward +y. Select the
    # exposed cube on the corresponding side of the pile.
    target_name=(min(initial,key=lambda n:initial[n][1]) if sign>0
                 else max(initial,key=lambda n:initial[n][1]))
    # Offset from the exposed cube; initial contact is with its neighbor and
    # the subsequent q1 sweep catches this cube tangentially.
    qapproach=Q.copy()
    y=float(initial[target_name][1] + sign*.042)
    s=settle_to(e,s,[-1.0,y,0],None,45,grip)
    s=settle_to(e,s,[-1.0,y,0],qapproach,130,grip)
    print("READY",sign,target_name,"initial",np.round(initial[target_name],4),"robot",np.round(robot(s),3))
    # Approach from left with q1=0 until first displacement, then freeze base.
    hitbase=None
    for k in range(50):
        a=action_to(s,[-.80,y,0],qapproach,grip); a[0]=min(a[0],.012)
        s,r,t,tr,i=e.step(a); cs=cubes(s)
        dif={n:float(np.linalg.norm(cs[n]-initial[n])) for n in cs}
        if max(dif.values())>.001:
            hitbase=robot(s)[:3].copy()
            print("HIT",sign,k,"reward",r,"base",np.round(hitbase,4),"diff",{n:round(v,4) for n,v in dif.items()},"cube",{n:np.round(v,4).tolist() for n,v in cs.items()})
            break
    if hitbase is None:
        print("NO_HIT",sign,"robot",np.round(robot(s),3)); e.close(); return
    # Sweep q1 while actively holding base at its contact pose.
    qtarget=Q.copy(); qtarget[0]=sign*.8
    before=cubes(s)
    hold_q=None; hold_y=None
    bin_name=("bin_green" if sign<0 else "bin_red")
    bin_xy=get(s,bin_name,["x","y"])[:2]
    for k in range(110):
        hold=hitbase.copy(); qnow=float(robot(s)[3])
        # Once lateral position reaches the bin, cancel further tangential y
        # motion with base translation while q1 continues pulling x left.
        if hold_q is not None:
            hold[1]=hold_y + sign*.88*(np.sin(abs(qnow))-np.sin(abs(hold_q)))
        s,r,t,tr,i=e.step(action_to(s,hold,qtarget,grip)); cs=cubes(s)
        delta={n:cs[n]-before[n] for n in cs}
        # Start braking ~4 cm early; the cube coasts while the base catches up.
        if hold_q is None and ((sign<0 and cs[target_name][1]>=bin_xy[1]-.04) or (sign>0 and cs[target_name][1]<=bin_xy[1]+.04)):
            hold_q=float(robot(s)[3]); hold_y=float(robot(s)[1])
            print("YHOLD",sign,k,"q",round(hold_q,4),"basey",round(hold_y,4),"cube",np.round(cs[target_name],4).tolist())
        if k%5==0:
            print("SWEEP",sign,k,"reward",r,"robot",np.round(robot(s),3),"delta",{n:np.round(v,4).tolist() for n,v in delta.items()})
        if np.linalg.norm(cs[target_name][:2]-bin_xy)<.045:
            print("NEAR_BIN",sign,k,"reward",r,"dist",round(float(np.linalg.norm(cs[target_name][:2]-bin_xy)),4)); break
    print("FINAL",sign,"q1",round(float(robot(s)[3]),4),"base",np.round(robot(s)[:3],4).tolist(),"target",target_name,"start",np.round(before[target_name],4).tolist(),"end",np.round(cubes(s)[target_name],4).tolist(),"all",{n:np.round(v,4).tolist() for n,v in cubes(s).items()},"bin_delta",{n:np.round(get(s,n,["x","y","z"])-v,4).tolist() for n,v in initial_bins.items()})
    e.close()

if __name__=="__main__":
    for sign in [-1]: lateral_trial(sign,0)
