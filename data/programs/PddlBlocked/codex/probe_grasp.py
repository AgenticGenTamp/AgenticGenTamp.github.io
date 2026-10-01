import numpy as np
from env_client import make_env


def dump(state, env, label=""):
    print("\n", label)
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        typ = next(t for t in env.observation_space.types if obj in state.get_objects(t))
        fs = env.observation_space.type_features[typ]
        vals = [state.get(obj, f) for f in fs]
        print(name, dict(zip(fs, np.round(vals, 4))))


if __name__ == "__main__":
    env = make_env()
    s, info = env.reset(seed=0)
    dump(s, env, "initial")
    print("info", info)
    env.close()


def robot_vals(state):
    r = state.get_object_from_name("robot")
    return np.array([state.get(r, f) for f in (
        "base_x", "base_y", "base_rot", "joint_1", "joint_2", "joint_3",
        "joint_4", "joint_5", "joint_6", "joint_7", "gripper_opening", "grasp_active")])


def goto(env, state, target):
    for _ in range(50):
        cur = robot_vals(state)[:10]
        d = np.clip(target - cur, -.2, .2)
        if np.max(np.abs(d)) < 1e-4:
            break
        a = np.zeros(11, np.float32); a[:10] = d
        state, _, done, trunc, _ = env.step(a)
    return state


def scan_default():
    env = make_env(); s, _ = env.reset(seed=0)
    q = robot_vals(s)[3:10]
    for rot in [0, .3, -.3, .6, -.6]:
      for x in np.arange(3.2, 4.01, .1):
       for y in np.arange(-.7, .71, .1):
        target = np.r_[x, y, rot, q]
        s = goto(env, s, target)
        a=np.zeros(11,np.float32); a[10]=-1
        s,_,_,_,_=env.step(a)
        if robot_vals(s)[11] > .5:
            print("GRASP", target, "actual", robot_vals(s))
            dump(s,env,"grasp")
            return
        a[10]=1; s,_,_,_,_=env.step(a)
    print("none")
    env.close()


def rot(axis, a):
    axis=np.asarray(axis,float); axis/=np.linalg.norm(axis)
    K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
    T=np.eye(4); T[:3,:3]=R; return T


def trans(x,y,z):
    T=np.eye(4); T[:3,3]=[x,y,z]; return T


def fk(base, q):
    T=trans(base[0],base[1],0)@rot([0,0,1],base[2])@trans(-.05,.188,1.04)
    T=T@rot([0,0,1],q[0])@trans(.1,0,0)@rot([0,1,0],q[1])@rot([1,0,0],q[2])
    T=T@trans(.4,0,0)@rot([0,1,0],q[3])@rot([1,0,0],q[4])
    T=T@trans(.321,0,0)@rot([0,1,0],q[5])@rot([1,0,0],q[6])@trans(.18,0,0)
    return T


def ik_test():
    from scipy.optimize import least_squares
    env=make_env(); s,_=env.reset(seed=0); ini=robot_vals(s); q0=ini[3:10]
    # Goal tool center at blocker; approach axis horizontal. Only position used.
    for base in ([3.5,0,0],[3.6,0,0],[3.5,-.3,0],[3.5,.3,0]):
      for name in ("blocker","green0"):
        o=s.get_object_from_name(name); p=np.array([s.get(o,f) for f in ("pose_x","pose_y","pose_z")])
        def fun(q):
            T=fk(base,q)
            # Horizontal tool x, also penalize tool z-axis's z to encourage vertical finger plane.
            return np.r_[10*(T[:3,3]-p), 2*T[2,0]]
        for k in range(8):
            start=q0+np.random.default_rng(k).normal(0,1,7)
            sol=least_squares(fun,start,max_nfev=1000).x
            pred=fk(base,sol)
            targ=np.r_[base,sol]
            ss=goto(env,s,targ)
            actual=robot_vals(ss)
            a=np.zeros(11,np.float32);a[10]=-1
            ss,_,_,_,_=env.step(a)
            print(name,"base",base,"err",np.linalg.norm(fun(sol)),"q",np.round(sol,3),"accepted",np.round(actual[:10],3),"g",robot_vals(ss)[11],"pred",np.round(pred[:3,3],3))
            if robot_vals(ss)[11]: dump(ss,env,"WIN"); return
            a[10]=1; ss,*_=env.step(a)
    env.close()


def solve_ik(base, p, direction=np.array([1.,0,0]), seed=0):
    from scipy.optimize import least_squares
    lo=np.array([-.715,-.524,-.8,-2.321,-np.pi,-2.094,-np.pi])
    hi=np.array([2.285,1.396,3.9,0,np.pi,0,np.pi])
    q0=np.array([.393,.333,0,-1.522,2.722,-1.22,-2.989])
    def fun(q):
        T=fk(base,q)
        return np.r_[10*(T[:3,3]-p), 3*np.cross(T[:3,0],direction), .03*(q-q0)]
    rng=np.random.default_rng(seed)
    best=None
    for k in range(5):
        start=q0 if k==0 else rng.uniform(lo,hi)
        z=least_squares(fun,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=400)
        if best is None or np.linalg.norm(fun(z.x))<best[0]: best=(np.linalg.norm(fun(z.x)),z.x)
    return best


def try_candidate(env, seed, base, qpre, qgoal):
    s,_=env.reset(seed=seed)
    # Close commanded all the time on approach catches any valid crossing.
    for target in (np.r_[base, qpre], np.r_[base,qgoal]):
      for _ in range(30):
        cur=robot_vals(s)[:10]; d=np.clip(target-cur,-.12,.12)
        a=np.zeros(11,np.float32);a[:10]=d;a[10]=-1
        s,_,_,_,_=env.step(a)
        if robot_vals(s)[11]: return True,s
        if np.max(np.abs(d))<1e-3: break
    return False,s


def calibrated_grid():
    env=make_env(); s,_=env.reset(seed=0)
    o=s.get_object_from_name('blocker'); real=np.array([s.get(o,f) for f in ('pose_x','pose_y','pose_z')])
    count=0
    for br in [-.6,-.4,-.2,0,.2]:
      for by in [-.3,-.1,.1,.3]:
       base=np.array([3.3,by,br])
       # model calibration offsets; pregrasp is 12cm backward along world x
       for dz in np.arange(-.25,.251,.05):
        for dy in np.arange(-.2,.201,.05):
         for dx in np.arange(-.2,.201,.05):
          modelp=real+np.array([dx,dy,dz])
          eg,qg=solve_ik(base,modelp,seed=count)
          ep,qp=solve_ik(base,modelp-np.array([.12,0,0]),seed=count)
          if eg>.2 or ep>.2: continue
          ok,ss=try_candidate(env,0,base,qp,qg);count+=1
          if count%100==0: print('tries',count,'base',base,'offset',dx,dy,dz,flush=True)
          if ok:
             print('SUCCESS',count,'base',base,'offset',dx,dy,dz,'qpre',qp,'qgoal',qg,'actual',robot_vals(ss),flush=True)
             dump(ss,env,'success');env.close();return
    print('none',count);env.close()
