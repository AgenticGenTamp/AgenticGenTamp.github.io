"""Narrow live probes for PR2Packed grasp configurations."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from env_client import make_env


def dump(seed=0):
    env = make_env()
    s, info = env.reset(seed=seed)
    print("max", env.max_steps, "info", info)
    print("types", [(t.name, [f.name if hasattr(f, 'name') else str(f) for f in env.observation_space.type_features[t]]) for t in env.observation_space.types])
    for name in s.get_object_names():
        o = s.get_object_from_name(name)
        typ = o.type.name
        fs = env.observation_space.type_features[o.type]
        vals = {getattr(f, 'name', str(f)): s.get(o, getattr(f, 'name', str(f))) for f in fs}
        print(name, typ, vals)
    env.close()


def val(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def move_delta(env, s, indices, targets, grip=0.0):
    """Reach absolute base/joint targets using legal relative increments."""
    features = {0: "base_x", 1: "base_y", 2: "base_rot",
                3: "joint_1", 4: "joint_2", 5: "joint_3", 6: "joint_4",
                7: "joint_5", 8: "joint_6", 9: "joint_7"}
    for _ in range(20):
        a = np.zeros(11, dtype=np.float32)
        done = True
        for i, target in zip(indices, targets):
            d = target - val(s, "robot", features[i])
            a[i] = np.clip(d, -0.2, 0.2)
            done &= abs(d) < 1e-4
        if done:
            break
        a[10] = grip
        s, _, term, trunc, _ = env.step(a)
        if term or trunc:
            return s
    return s


def scan(seed=0, qoffset=None, max_points=450):
    """Scan base translations under a fixed arm posture and close at each point."""
    env = make_env()
    s, info = env.reset(seed=seed)
    q0 = [val(s, "robot", f"joint_{i}") for i in range(1, 8)]
    if qoffset is not None:
        qt = [x + y for x, y in zip(q0, qoffset)]
        s = move_delta(env, s, list(range(3, 10)), qt, grip=1.0)
    # Traverse continuous rows cheaply. The actual base values expose rejected moves.
    xs = np.linspace(-1.38, -0.58, 17)
    ys = np.linspace(-0.95, 0.95, 39)
    count = 0
    for ri, x in enumerate(xs):
        row = ys if ri % 2 == 0 else ys[::-1]
        for y in row:
            if count >= max_points:
                env.close(); return False
            s = move_delta(env, s, [0, 1], [x, y], grip=1.0)
            # Explicit close even if the requested base motion was rejected.
            a = np.zeros(11, dtype=np.float32); a[10] = -1.0
            s, _, term, trunc, _ = env.step(a)
            count += 1
            if val(s, "robot", "grasp_active") > 0.5:
                q = [val(s, "robot", f"joint_{i}") for i in range(1, 8)]
                base = [val(s, "robot", z) for z in ("base_x", "base_y", "base_rot")]
                held = [n for n in s.get_object_names()
                        if n.startswith("block") and val(s, n, "grasp_active") > .5]
                tf = [val(s, "robot", "grasp_tf_" + z) for z in ("x","y","z","qx","qy","qz","qw")]
                print("SUCCESS", seed, "offset", qoffset, "count", count,
                      "base", base, "q", q, "held", held, "tf", tf, flush=True)
                env.close(); return True
            # Re-open for the next trial.
            a[10] = 1.0
            s, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.close(); return False
    env.close()
    print("failed", seed, qoffset, count, flush=True)
    return False


def targeted(seed=0, tool_xy=(.602, .250), z_deltas=(0.0,)):
    """Test FK-predicted base alignments against every block."""
    for zdelta in z_deltas:
        env = make_env(); s, _ = env.reset(seed=seed)
        blocks = []
        for n in s.get_object_names():
            if n.startswith("block"):
                blocks.append((n, val(s,n,"pose_x"), val(s,n,"pose_y")))
        q2 = val(s,"robot","joint_2") + zdelta
        s = move_delta(env, s, [4], [q2], grip=1.0)
        for n, bx, by in blocks:
            tx, ty = bx-tool_xy[0], by-tool_xy[1]
            s = move_delta(env, s, [0,1], [tx,ty], grip=1.0)
            a=np.zeros(11,dtype=np.float32); a[10]=-1
            s,_,term,trunc,_=env.step(a)
            print("trial", n, "zd", zdelta, "requested",tx,ty,"actual",
                  val(s,"robot","base_x"),val(s,"robot","base_y"),
                  "grasp",val(s,"robot","grasp_active"), flush=True)
            if val(s,"robot","grasp_active")>.5:
                q=[val(s,"robot",f"joint_{i}") for i in range(1,8)]
                tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
                print("SUCCESS base/q/tf",tx,ty,q,tf,flush=True)
                env.close(); return True
            a[10]=1; s,_,_,_,_=env.step(a)
        env.close()
    return False


def local_xy(seed=0, radius=.16, step=.02):
    """Dense local scan around the FK prediction for the reachable block."""
    env=make_env(); s,_=env.reset(seed=seed)
    blocks=[(n,val(s,n,"pose_x"),val(s,n,"pose_y")) for n in s.get_object_names()
            if n.startswith("block")]
    # Prefer block whose required base x is collision-free.
    n,bx,by=min(blocks,key=lambda b: abs((b[1]-.602)-(-.68)))
    center=(bx-.602,by-.250)
    offsets=np.arange(-radius,radius+step/2,step)
    count=0
    for ix,dx in enumerate(offsets):
        dys=offsets if ix%2==0 else offsets[::-1]
        for dy in dys:
            s=move_delta(env,s,[0,1],[center[0]+dx,center[1]+dy],grip=1)
            a=np.zeros(11,dtype=np.float32);a[10]=-1
            s,_,_,_,_=env.step(a);count+=1
            if val(s,"robot","grasp_active")>.5:
                base=[val(s,"robot",z) for z in ("base_x","base_y","base_rot")]
                q=[val(s,"robot",f"joint_{i}") for i in range(1,8)]
                tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
                print("SUCCESS local",n,count,"base",base,"q",q,"tf",tf,flush=True)
                env.close();return True
            a[10]=1;s,_,_,_,_=env.step(a)
    print("local failure",n,count,center,flush=True);env.close();return False


def ik_grasp(seed=0):
    """Execute a coupled 7-DOF vertical descent preserving tool orientation."""
    env=make_env();s,_=env.reset(seed=seed)
    q0=np.array([val(s,"robot",f"joint_{i}") for i in range(1,8)])
    dq=np.array([.03619296,.20777813,.13319970,-.19220320,
                 .16886771,.16129730,-.06069881])
    s=move_delta(env,s,list(range(3,10)),q0+dq,grip=1)
    print("q achieved",[val(s,"robot",f"joint_{i}") for i in range(1,8)],flush=True)
    blocks=[(n,val(s,n,"pose_x"),val(s,n,"pose_y")) for n in s.get_object_names()
            if n.startswith("block")]
    for n,bx,by in blocks:
        tx,ty=bx-.5241268,by-.1873725
        s=move_delta(env,s,[0,1],[tx,ty],grip=1)
        a=np.zeros(11,dtype=np.float32);a[10]=-1;s,_,_,_,_=env.step(a)
        print("IK trial",n,"target",tx,ty,"actual",val(s,"robot","base_x"),
              val(s,"robot","base_y"),"grasp",val(s,"robot","grasp_active"),flush=True)
        if val(s,"robot","grasp_active")>.5:
            tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
            print("SUCCESS IK",n,"base",tx,ty,"q",list(q0+dq),"tf",tf,flush=True)
            env.close();return True
        a[10]=1;s,_,_,_,_=env.step(a)
    env.close();return False


def fk(q):
    axes=np.eye(3)[[2,1,0,1,0,1,0]];lens=[0,.4,0,.321,0,.1,.18]
    T=np.eye(4)
    for qi,axis,length in zip(q,axes,lens):
        T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(qi*axis).as_matrix()
        T[:3,3]+=T[:3,:3]@np.array([length,0,0])
    return T


def solve_descent(q0,dz):
    T0=fk(q0);target=T0[:3,3]+[0,0,dz]
    def fun(q):
        T=fk(q)
        orient=Rotation.from_matrix(T0[:3,:3].T@T[:3,:3]).as_rotvec()
        return np.r_[12*(T[:3,3]-target),orient,.01*(q-q0)]
    return least_squares(fun,q0,max_nfev=1000).x


def descent_sweep(seed=0):
    """Sweep vertical tool height with FK-preserving coupled IK."""
    for dz in (-.10,-.13,-.15,-.17,-.19,-.21,-.23,-.25,-.28):
        env=make_env();s,_=env.reset(seed=seed)
        q0=np.array([val(s,"robot",f"joint_{i}") for i in range(1,8)])
        q=solve_descent(q0,dz)
        blocks=[(n,val(s,n,"pose_x"),val(s,n,"pose_y")) for n in s.get_object_names()
                if n.startswith("block")]
        # seed zero's leftmost block allows direct base access.
        n,bx,by=min(blocks,key=lambda b:b[1])
        rel=fk(q)[:3,3];tx=bx-(.05+rel[0]);ty=by-(.188+rel[1])
        s=move_delta(env,s,list(range(3,10)),q,grip=1)
        qerr=max(abs(val(s,"robot",f"joint_{i}")-q[i-1]) for i in range(1,8))
        s=move_delta(env,s,[0,1],[tx,ty],grip=1)
        a=np.zeros(11,dtype=np.float32);a[10]=-1;s,_,_,_,_=env.step(a)
        print("sweep dz",dz,"qerr",qerr,"target",tx,ty,"actual",
              val(s,"robot","base_x"),val(s,"robot","base_y"),"grasp",
              val(s,"robot","grasp_active"),flush=True)
        if val(s,"robot","grasp_active")>.5:
            tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
            print("SUCCESS SWEEP dz",dz,"block",n,"base",tx,ty,"q",q.tolist(),"tf",tf,flush=True)
            env.close();return True
        env.close()
    return False


def xyz_local(seed=0, dz=-.18):
    """Dense lateral scan at a plausible IK-derived grasp height."""
    env=make_env();s,_=env.reset(seed=seed)
    q0=np.array([val(s,"robot",f"joint_{i}") for i in range(1,8)])
    q=solve_descent(q0,dz);s=move_delta(env,s,list(range(3,10)),q,grip=1)
    blocks=[(n,val(s,n,"pose_x"),val(s,n,"pose_y")) for n in s.get_object_names()
            if n.startswith("block")]
    n,bx,by=min(blocks,key=lambda b:b[1]); rel=fk(q)[:3,3]
    blockyaw=2*np.arctan2(val(s,n,"pose_qz"),val(s,n,"pose_qw"))
    R=fk(q)[:3,:3];gapyaw=np.arctan2(R[1,1],R[0,1])
    ds=[(gapyaw-e+np.pi)%(2*np.pi)-np.pi for e in (blockyaw,blockyaw+np.pi/2)]
    q[6]+=min(ds,key=abs)
    s=move_delta(env,s,[9],[q[6]],grip=1)
    center=(bx-(.05+rel[0]),by-(.188+rel[1]))
    offsets=np.arange(-.09,.091,.015)
    for ix,dx in enumerate(offsets):
        for dy in (offsets if ix%2==0 else offsets[::-1]):
            s=move_delta(env,s,[0,1],[center[0]+dx,center[1]+dy],grip=1)
            a=np.zeros(11,dtype=np.float32);a[10]=-1;s,_,_,_,_=env.step(a)
            if val(s,"robot","grasp_active")>.5:
                base=[val(s,"robot",z) for z in ("base_x","base_y","base_rot")]
                tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
                print("SUCCESS XYZ dz",dz,"block",n,"base",base,"q",q.tolist(),"tf",tf,flush=True)
                env.close();return True
            a[10]=1;s,_,_,_,_=env.step(a)
    print("xyz local failed",dz,center,flush=True);env.close();return False


def yaw_sweep(seed=0):
    """Align the fingers with block edges, then sweep the vertical grasp height."""
    for dz in (-.12,-.14,-.16,-.18,-.20,-.21):
        env=make_env();s,_=env.reset(seed=seed)
        blocks=[(n,val(s,n,"pose_x"),val(s,n,"pose_y")) for n in s.get_object_names()
                if n.startswith("block")]
        n,bx,by=min(blocks,key=lambda b:b[1])
        bzq=val(s,n,"pose_qz");bwq=val(s,n,"pose_qw")
        blockyaw=2*np.arctan2(bzq,bwq)
        q0=np.array([val(s,"robot",f"joint_{i}") for i in range(1,8)])
        q=solve_descent(q0,dz)
        R=fk(q)[:3,:3]; gapyaw=np.arctan2(R[1,1],R[0,1])
        # Wrist roll rotates around downward approach: gap yaw decreases with dq.
        candidates=[]
        for edge in (blockyaw,blockyaw+np.pi/2):
            d=(gapyaw-edge+np.pi)%(2*np.pi)-np.pi
            candidates.append(d)
        dq7=min(candidates,key=abs);q[6]+=dq7
        rel=fk(q)[:3,3];tx=bx-(.05+rel[0]);ty=by-(.188+rel[1])
        s=move_delta(env,s,list(range(3,10)),q,grip=1)
        s=move_delta(env,s,[0,1],[tx,ty],grip=1)
        a=np.zeros(11,dtype=np.float32);a[10]=-1;s,_,_,_,_=env.step(a)
        print("yaw trial",dz,"yaw",blockyaw,"dq7",dq7,"base",tx,ty,
              "actual",val(s,"robot","base_x"),val(s,"robot","base_y"),
              "grasp",val(s,"robot","grasp_active"),flush=True)
        if val(s,"robot","grasp_active")>.5:
            tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
            print("SUCCESS YAW",n,"dz",dz,"base",tx,ty,"q",q.tolist(),"tf",tf,flush=True)
            env.close();return True
        env.close()
    return False


def overhead_descent(seed=0):
    """Align overhead while high, then descend vertically around the block."""
    for dz in (-.19,-.21,-.22,-.23,-.24,-.25,-.27,-.30,-.33,-.36,-.40):
        env=make_env();s,_=env.reset(seed=seed)
        blocks=[(n,val(s,n,"pose_x"),val(s,n,"pose_y")) for n in s.get_object_names()
                if n.startswith("block")]
        n,bx,by=min(blocks,key=lambda b:b[1])
        blockyaw=2*np.arctan2(val(s,n,"pose_qz"),val(s,n,"pose_qw"))
        q0=np.array([val(s,"robot",f"joint_{i}") for i in range(1,8)])
        q=solve_descent(q0,dz)
        R=fk(q)[:3,:3];gapyaw=np.arctan2(R[1,1],R[0,1])
        ds=[(gapyaw-e+np.pi)%(2*np.pi)-np.pi for e in (blockyaw,blockyaw+np.pi/2)]
        q[6]+=min(ds,key=abs)
        rel=fk(q)[:3,3];tx=bx-(.05+rel[0]);ty=by-(.188+rel[1])
        # Move the base under the high gripper before lowering it.
        s=move_delta(env,s,[0,1],[tx,ty],grip=1)
        s=move_delta(env,s,list(range(3,10)),q,grip=1)
        qerr=max(abs(val(s,"robot",f"joint_{i}")-q[i-1]) for i in range(1,8))
        a=np.zeros(11,dtype=np.float32);a[10]=-1;s,_,_,_,_=env.step(a)
        print("overhead",dz,"qerr",qerr,"base",val(s,"robot","base_x"),
              val(s,"robot","base_y"),"grasp",val(s,"robot","grasp_active"),flush=True)
        if val(s,"robot","grasp_active")>.5:
            tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
            print("SUCCESS OVERHEAD",n,"dz",dz,"base",tx,ty,"q",q.tolist(),"tf",tf,flush=True)
            env.close();return True
        env.close()
    return False


def incremental_descent(seed=0):
    """Follow many IK waypoints and try closing at each height."""
    env=make_env();s,_=env.reset(seed=seed)
    blocks=[(n,val(s,n,"pose_x"),val(s,n,"pose_y")) for n in s.get_object_names()
            if n.startswith("block")]
    n,bx,by=min(blocks,key=lambda b:b[1])
    blockyaw=2*np.arctan2(val(s,n,"pose_qz"),val(s,n,"pose_qw"))
    q0=np.array([val(s,"robot",f"joint_{i}") for i in range(1,8)])
    R=fk(q0)[:3,:3];gapyaw=np.arctan2(R[1,1],R[0,1])
    ds=[(gapyaw-e+np.pi)%(2*np.pi)-np.pi for e in (blockyaw,blockyaw+np.pi/2)]
    yaw_delta=min(ds,key=abs)
    # XY is nearly invariant, align while safely overhead.
    rel=fk(q0)[:3,3];tx=bx-(.05+rel[0]);ty=by-(.188+rel[1])
    s=move_delta(env,s,[0,1],[tx,ty],grip=1)
    for dz in np.arange(-.08,-.401,-.005):
        q=solve_descent(q0,float(dz));q[6]+=yaw_delta
        s=move_delta(env,s,list(range(3,10)),q,grip=1)
        qerr=max(abs(val(s,"robot",f"joint_{i}")-q[i-1]) for i in range(1,8))
        a=np.zeros(11,dtype=np.float32);a[10]=-1;s,_,_,_,_=env.step(a)
        if val(s,"robot","grasp_active")>.5:
            tf=[val(s,"robot","grasp_tf_"+z) for z in ("x","y","z","qx","qy","qz","qw")]
            actualq=[val(s,"robot",f"joint_{i}") for i in range(1,8)]
            print("SUCCESS INCREMENTAL",n,"dz",dz,"base",tx,ty,"q",actualq,"tf",tf,flush=True)
            env.close();return True
        a[10]=1;s,_,_,_,_=env.step(a)
        if int(round((-dz-.08)/.005))%10==0:
            print("incremental",dz,"qerr",qerr,flush=True)
    env.close();return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        dump(0)
    else:
        # First scan current posture; argument selects a shoulder/elbow height family.
        k = int(sys.argv[1])
        variants = [
            [0,0,0,0,0,0,0], [0,-.3,0,0,0,0,0], [0,.3,0,0,0,0,0],
            [0,0,0,-.3,0,0,0], [0,0,0,.3,0,0,0],
            [0,-.3,0,.3,0,0,0], [0,.3,0,-.3,0,0,0],
        ]
        if k == 99:
            targeted(0, z_deltas=(0,.1,-.1,.2,-.2,.3,-.3))
        elif k == 98:
            local_xy(0)
        elif k == 97:
            ik_grasp(0)
        elif k == 96:
            descent_sweep(0)
        elif k <= 95:
            if k == 90:
                yaw_sweep(0)
            elif k == 89:
                overhead_descent(0)
            elif k == 88:
                incremental_descent(0)
            else:
                xyz_local(0, -.12-.02*(95-k))
        else:
            scan(0, variants[k], max_points=450)
