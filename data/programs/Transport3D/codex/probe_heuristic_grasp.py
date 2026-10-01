"""Targeted grasp/push experiments; not imported by the policy."""
from env_client import make_env
import numpy as np


def get(s, name, feat):
    return float(s.get(s.get_object_from_name(name), feat))


def act(env, s, vals):
    a = np.zeros(11, dtype=np.float32)
    for i, v in vals.items():
        a[i] = v
    return env.step(a)[0]


def scan(seed, target, q_delta=None, spacing=0.03):
    env = make_env(); s, _ = env.reset(seed=seed)
    tx, ty = get(s,target,"pose_x"), get(s,target,"pose_y")
    # Move base to left end, aligned laterally, using bounded exact deltas.
    bx, by = get(s,"robot","pos_base_x"), get(s,"robot","pos_base_y")
    goalx, goaly = tx - 1.0, ty
    while abs(goalx-bx)>1e-5 or abs(goaly-by)>1e-5:
        dx=np.clip(goalx-bx,-.2,.2); dy=np.clip(goaly-by,-.2,.2)
        s=act(env,s,{0:dx,1:dy}); bx=get(s,"robot","pos_base_x"); by=get(s,"robot","pos_base_y")
    if q_delta:
        for axis, total in q_delta.items():
            while abs(total)>1e-5:
                d=float(np.clip(total,-.2,.2)); s=act(env,s,{axis:d}); total-=d
    for k in range(int(2.0/spacing)+1):
        # alternate open/close at each position, then advance
        s=act(env,s,{10:1.0}); s=act(env,s,{10:-1.0})
        if get(s,"robot","grasp_active")>.5 or get(s,target,"grasp_active")>.5:
            print("FOUND",seed,target,"base",get(s,"robot","pos_base_x"),get(s,"robot","pos_base_y"),
                  "q",[round(get(s,"robot",f"joint_{i}"),3) for i in range(1,8)],
                  "tf",[get(s,"robot",f"grasp_tf_{c}") for c in "xyz"])
            # Test teleport through base motion.
            before=[get(s,target,"pose_"+c) for c in "xyz"]
            s=act(env,s,{0:.2,1:.2})
            after=[get(s,target,"pose_"+c) for c in "xyz"]
            print("MOVE ATTACHED",before,after)
            env.close(); return True
        s=act(env,s,{0:spacing})
    print("none",seed,target,"qdelta",q_delta)
    env.close(); return False


def push(seed, target):
    env=make_env(); s,_=env.reset(seed=seed)
    tx,ty=get(s,target,"pose_x"),get(s,target,"pose_y")
    # Sweep base center across target along x, at target y.
    for gx,gy in [(tx-.8,ty),(tx+.8,ty)]:
        while abs(gx-get(s,"robot","pos_base_x"))>1e-4 or abs(gy-get(s,"robot","pos_base_y"))>1e-4:
            s=act(env,s,{0:np.clip(gx-get(s,"robot","pos_base_x"),-.2,.2),1:np.clip(gy-get(s,"robot","pos_base_y"),-.2,.2)})
    print("PUSH",target,"target now",[get(s,target,"pose_"+c) for c in "xyz"],"base",get(s,"robot","pos_base_x"),get(s,"robot","pos_base_y"))
    env.close()


def random_search(seed=0, target="box0", search_seed=0):
    rng=np.random.default_rng(search_seed)
    env=make_env(); s,_=env.reset(seed=seed)
    tx,ty=get(s,target,"pose_x"),get(s,target,"pose_y")
    # Realized values may clip at joint limits; report those on success.
    for attempt in range(52):
        desired_base=np.array([tx+rng.uniform(-1.15,1.15),ty+rng.uniform(-1.15,1.15),rng.uniform(-3.14,3.14)])
        desired_q=rng.uniform(-3.1,3.1,7)
        # Move all coordinates toward random target concurrently while open.
        for _ in range(32):
            cur=np.array([get(s,"robot","pos_base_x"),get(s,"robot","pos_base_y"),get(s,"robot","pos_base_rot")])
            cq=np.array([get(s,"robot",f"joint_{i}") for i in range(1,8)])
            a={0:np.clip(desired_base[0]-cur[0],-.2,.2),1:np.clip(desired_base[1]-cur[1],-.2,.2),
               2:np.clip(desired_base[2]-cur[2],-.2,.2),10:1.0}
            for j in range(7): a[3+j]=np.clip(desired_q[j]-cq[j],-.2,.2)
            s=act(env,s,a)
        s=act(env,s,{10:-1.0})
        if get(s,"robot","grasp_active")>.5:
            q=[get(s,"robot",f"joint_{i}") for i in range(1,8)]
            b=[get(s,"robot","pos_base_x"),get(s,"robot","pos_base_y"),get(s,"robot","pos_base_rot")]
            off=[b[0]-tx,b[1]-ty]
            print("RFOUND",search_seed,attempt,target,"base",b,"offset",off,"q",q,"obj",[get(s,target,"pose_"+c) for c in "xyz"],"tf",[get(s,"robot",f"grasp_tf_{c}") for c in "xyz"])
            env.close(); return True
        if attempt>=29:  # leave room under max_steps
            break
    print("rnone",search_seed)
    env.close(); return False


def panda_fk(q):
    # Standard Franka DH hypothesis; useful for structured posture search.
    a=[0,0,0,.0825,-.0825,0,.088]; d=[.333,0,.316,0,.384,0,.107]
    al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2]
    T=np.eye(4)
    for qi,ai,di,aa in zip(q,a,d,al):
        c,ss=np.cos(qi),np.sin(qi); ca,sa=np.cos(aa),np.sin(aa)
        T=T@np.array([[c,-ss*ca,ss*sa,ai*c],[ss,c*ca,-c*sa,ai*ss],[0,sa,ca,di],[0,0,0,1.]])
    return T


def ik_search(seed=0,target="box0"):
    from scipy.optimize import least_squares
    env=make_env(); s,_=env.reset(seed=seed)
    tx,ty=get(s,target,"pose_x"),get(s,target,"pose_y")
    q0=np.array([0,-.35,-np.pi,-2.5,0,-.87,np.pi/2])
    for zrel in [-.35,-.25,-.15,-.05,.05,.15,.25,.35]:
      for orient_axis in [0,2]:
        def residual(q):
            T=panda_fk(q)
            ori=T[:3,orient_axis]-np.array([0,0,-1.])
            return np.r_[3*(T[:3,3]-np.array([.4,0,zrel])),ori]
        sol=least_squares(residual,q0,bounds=(-3.15,3.15),max_nfev=1000).x
        ee=panda_fk(sol)[:3,3]
        # Command posture and predicted base location concurrently.
        goal=np.array([tx-ee[0],ty-ee[1],0.])
        for _ in range(35):
            cur=np.array([get(s,"robot","pos_base_x"),get(s,"robot","pos_base_y"),get(s,"robot","pos_base_rot")])
            cq=np.array([get(s,"robot",f"joint_{i}") for i in range(1,8)])
            aa={0:np.clip(goal[0]-cur[0],-.2,.2),1:np.clip(goal[1]-cur[1],-.2,.2),2:np.clip(-cur[2],-.2,.2),10:1.}
            for j in range(7): aa[3+j]=np.clip(sol[j]-cq[j],-.2,.2)
            s=act(env,s,aa)
        s=act(env,s,{10:-1.})
        if get(s,"robot","grasp_active")>.5:
            print("IKFOUND",zrel,orient_axis,"q",sol.tolist(),"ee",ee.tolist(),"base",goal.tolist(),"tf",[get(s,"robot",f"grasp_tf_{c}") for c in "xyz"])
            env.close(); return
    print("iknone");env.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv)==2 and sys.argv[1]=="ik":
        ik_search()
    elif len(sys.argv) == 2:
        random_search(search_seed=int(sys.argv[1]))
    elif len(sys.argv) == 3:
        axis, delta = int(sys.argv[1]), float(sys.argv[2])
        scan(0, "box0", {axis: delta}, spacing=.05)
    else:
        push(0,"cube1")
        scan(0,"cube1")
        scan(0,"box0")
