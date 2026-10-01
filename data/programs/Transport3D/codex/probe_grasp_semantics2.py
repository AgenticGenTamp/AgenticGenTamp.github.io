"""Independent probes for gripper command and attachment semantics."""
import argparse
import math

import numpy as np

from env_client import make_env


def get(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def step(env, state, action):
    state, reward, terminated, truncated, _ = env.step(
        np.asarray(action, dtype=np.float32))
    return state, terminated or truncated


def summary(state, target):
    return {
        "base": [round(get(state, "robot", f), 4) for f in
                 ("pos_base_x", "pos_base_y", "pos_base_rot")],
        "finger": get(state, "robot", "finger_state"),
        "robot_grasp": get(state, "robot", "grasp_active"),
        "object_grasp": get(state, target, "grasp_active"),
        "object": [round(get(state, target, "pose_" + c), 4) for c in "xyz"],
    }


def command_probe(seed):
    """Show threshold/sign and whether holding or pulsing commands differs."""
    for command in (-1.0, -0.6, -0.49, 0.0, 0.49, 0.6, 1.0):
        env = make_env(); state, _ = env.reset(seed=seed)
        target = "box0"
        values = []
        for _ in range(4):
            action = np.zeros(11); action[10] = command
            state, _ = step(env, state, action)
            values.append(get(state, "robot", "finger_state"))
        print("command", command, "finger", values)
        env.close()


def direct_probe(seed, target):
    """Place base at/near target and test close sequencing variants."""
    for ox, oy in ((0,0),(-.2,0),(.2,0),(0,-.2),(0,.2),(-.4,0),(.4,0)):
      for sequence in ((-1.,), (1.,-1.), (-1.,1.,-1.),
                       (1.,1.,-1.,-1.), (-1.,-1.,-1.)):
        env=make_env(); state,_=env.reset(seed=seed)
        gx=get(state,target,"pose_x")+ox; gy=get(state,target,"pose_y")+oy
        for _ in range(12):
            action=np.zeros(11); action[0]=np.clip(gx-get(state,"robot","pos_base_x"),-.2,.2)
            action[1]=np.clip(gy-get(state,"robot","pos_base_y"),-.2,.2)
            state,done=step(env,state,action)
        for command in sequence:
            action=np.zeros(11);action[10]=command;state,done=step(env,state,action)
        if get(state,"robot","grasp_active")>.5:
            print("DIRECT HIT",[ox,oy],sequence,summary(state,target));env.close();return
        print("direct miss",[ox,oy],sequence)
        env.close()


def range_probe(seed):
    """Empirically find each joint's realized upper value from its reset lower limit."""
    for axis in range(3,10):
        env=make_env();state,_=env.reset(seed=seed)
        lo=get(state,"robot",f"joint_{axis-2}")
        for _ in range(40):
            action=np.zeros(11); action[axis]=.2; state,_=step(env,state,action)
        hi=get(state,"robot",f"joint_{axis-2}")
        print("range",axis-2,lo,hi)
        env.close()


def broad_probe(seed, target):
    """Search truly broad joint ranges, including positive elbow/wrist values."""
    rng=np.random.default_rng(90210)
    env=make_env();state,_=env.reset(seed=seed)
    tx,ty=get(state,target,"pose_x"),get(state,target,"pose_y")
    configs=[]
    # First vary the three bounded pitch-like joints, then space-fill all joints.
    configs += [np.array([0., q2, -np.pi, q4, 0., q6, np.pi/2])
                for q2,q4,q6 in ((-.35,0.,.6),(.15,0.,.6),(.65,0.,.6),
                                  (-.35,1.5,1.5),(.15,1.5,1.5),(.65,1.5,1.5))]
    while len(configs)<11:
        configs.append(np.array([rng.uniform(0,6.28),rng.uniform(-.35,.65),
                                  rng.uniform(-3.14,3.14),rng.uniform(-2.5,2.66),
                                  rng.uniform(0,6.28),rng.uniform(-.87,2.23),
                                  rng.uniform(1.57,7.85)]))
    offsets=np.linspace(-.65,.65,5)
    steps=0
    for ci,q in enumerate(configs):
        # Move posture and base together, held open.
        gx,gy=tx+offsets[0],ty+offsets[0]
        for _ in range(34):
            action=np.zeros(11);action[10]=1.
            action[0]=np.clip(gx-get(state,"robot","pos_base_x"),-.2,.2)
            action[1]=np.clip(gy-get(state,"robot","pos_base_y"),-.2,.2)
            for j in range(7):action[3+j]=np.clip(q[j]-get(state,"robot",f"joint_{j+1}"),-.2,.2)
            state,done=step(env,state,action);steps+=1
            if max(abs(action[0]),abs(action[1]),np.max(np.abs(action[3:10])))<.01:break
        for iy,oy in enumerate(offsets):
            row=offsets if iy%2==0 else offsets[::-1]
            for ox in row:
                gx,gy=tx+ox,ty+oy
                while max(abs(gx-get(state,"robot","pos_base_x")),abs(gy-get(state,"robot","pos_base_y")))>.015:
                    action=np.zeros(11);action[10]=1.;action[0]=np.clip(gx-get(state,"robot","pos_base_x"),-.2,.2);action[1]=np.clip(gy-get(state,"robot","pos_base_y"),-.2,.2)
                    state,done=step(env,state,action);steps+=1
                    if done:break
                action=np.zeros(11);action[10]=1.;state,done=step(env,state,action);steps+=1
                action=np.zeros(11);action[10]=-1.;action[0]=.01;action[6]=.01
                state,done=step(env,state,action);steps+=1
                if get(state,"robot","grasp_active")>.5:
                    print("BROAD HIT",ci,"q",q.tolist(),"offset",[ox,oy],summary(state,target),"steps",steps)
                    env.close();return
                if done:
                    print("broad ended",ci,"steps",steps);env.close();return
        print("broad miss",ci,"q",np.round(q,2).tolist(),"steps",steps,flush=True)
    env.close()


def seed_sweep(start, count):
    """Exploit randomized object placement to find contact at reset posture."""
    for seed in range(start,start+count):
        env=make_env();state,_=env.reset(seed=seed)
        names=[n for n in state.get_object_names() if n not in ("robot","table")]
        # Explicit open then close, moving slightly on the closing transition.
        action=np.zeros(11);action[10]=1.;state,_=step(env,state,action)
        action=np.zeros(11);action[10]=-1.;action[0]=.01;action[3]=.01
        state,_=step(env,state,action)
        held=[n for n in names if get(state,n,"grasp_active")>.5]
        if get(state,"robot","grasp_active")>.5 or held:
            print("SEED HIT",seed,held,{n:[get(state,n,"pose_"+c) for c in "xyz"] for n in names},summary(state,held[0] if held else names[0]))
            env.close();return
        if seed%25==0:print("seed miss",seed,flush=True)
        env.close()
    print("seed sweep miss",start,count)


def replay_route(seed, target, keep):
    """Replay the independently discovered successful suffix relative to target."""
    lo=np.array([0.,-.35,-math.pi,-2.5,0.,-.87,math.pi/2])
    hi=lo+np.array([5.2,2.41,2.058,2.66,5.2,2.23,6.77])
    rng=np.random.default_rng(0);route=[]
    for _ in range(28):
        q=rng.uniform(lo,hi);r=rng.uniform(.05,.95);ang=rng.uniform(-math.pi,math.pi)
        route.append((q,r,ang,rng.uniform(-math.pi,math.pi)))
    env=make_env();state,_=env.reset(seed=seed)
    tx,ty=get(state,target,"pose_x"),get(state,target,"pose_y")
    for ri,(q,r,ang,rot) in enumerate(route[-keep:],28-keep):
        goal=np.array([tx+r*math.cos(ang),ty+r*math.sin(ang),rot])
        for grip,repeats in ((1.,35),(-1.,2)):
            for rep in range(repeats):
                action=np.zeros(11);action[:3]=np.clip(goal-[get(state,"robot",f) for f in ("pos_base_x","pos_base_y","pos_base_rot")],-.2,.2)
                action[3:10]=np.clip(q-[get(state,"robot",f"joint_{j}") for j in range(1,8)],-.2,.2);action[10]=grip
                if grip < 0 and ri == 27 and rep == 0:
                    action[0] += .01; action[3] += .01
                state,done=step(env,state,action)
                if grip < 0 and ri == 27 and rep == 0:
                    print("simultaneous close first-frame active",get(state,"robot","grasp_active"),flush=True)
        print("route",ri,"robot",get(state,"robot","grasp_active"),"object",get(state,target,"grasp_active"),"base-offset",[get(state,"robot","pos_base_x")-tx,get(state,"robot","pos_base_y")-ty],flush=True)
        if get(state,"robot","grasp_active")>.5:
            before=[get(state,target,"pose_"+c) for c in "xyz"]
            action=np.zeros(11);action[:2]=(.2,.1);action[10]=-1.;state,_=step(env,state,action)
            print("ROUTE HIT",target,"before",before,"after",[get(state,target,"pose_"+c) for c in "xyz"],summary(state,target))
            action=np.zeros(11);action[10]=1.;action[0]=-.01;action[3]=.01;state,_=step(env,state,action)
            released=[get(state,target,"pose_"+c) for c in "xyz"]
            action=np.zeros(11);action[:2]=(-.2,-.1);action[10]=1.;state,_=step(env,state,action)
            print("RELEASE",summary(state,target),"object fixed",released,"=>",[get(state,target,"pose_"+c) for c in "xyz"]);break
    env.close()


def cube_refine(seed, target):
    """Perturb the box-grasp posture to seek the cube's 7.5cm lower center."""
    lo=np.array([0.,-.35,-math.pi,-2.5,0.,-.87,math.pi/2]);hi=lo+np.array([5.2,2.41,2.058,2.66,5.2,2.23,6.77])
    rng=np.random.default_rng(0);route=[]
    for _ in range(28):
        q=rng.uniform(lo,hi);r=rng.uniform(.05,.95);ang=rng.uniform(-math.pi,math.pi);route.append((q,r,ang,rng.uniform(-math.pi,math.pi)))
    final=route[-1];q0=final[0]
    # Even numbered joints dominate elevation in a serial arm.
    deltas=[(1,-.25),(1,.25),(3,-.25),(3,.25),(5,-.25),(5,.25)]
    for ji,dq in deltas:
        env=make_env();state,_=env.reset(seed=seed);tx,ty=get(state,target,"pose_x"),get(state,target,"pose_y")
        # Reproduce approach history through the preceding three configurations.
        for q,r,ang,rot in route[-4:-1]:
            goal=np.array([tx+r*math.cos(ang),ty+r*math.sin(ang),rot])
            for _ in range(35):
                action=np.zeros(11);action[:3]=np.clip(goal-[get(state,"robot",f) for f in ("pos_base_x","pos_base_y","pos_base_rot")],-.2,.2);action[3:10]=np.clip(q-[get(state,"robot",f"joint_{j}") for j in range(1,8)],-.2,.2);action[10]=1.;state,_=step(env,state,action)
        q=q0.copy();q[ji]+=dq;base0=np.array([tx-.42818514,ty+.05494344,final[3]])
        # Dense horizontal compensation around the known box offset.
        vals=np.linspace(-.18,.18,9)
        for iy,dy in enumerate(vals):
            xs=vals if iy%2==0 else vals[::-1]
            for dx in xs:
                goal=base0+np.array([dx,dy,0.])
                for _ in range(8):
                    action=np.zeros(11);action[:3]=np.clip(goal-[get(state,"robot",f) for f in ("pos_base_x","pos_base_y","pos_base_rot")],-.2,.2);action[3:10]=np.clip(q-[get(state,"robot",f"joint_{j}") for j in range(1,8)],-.2,.2);action[10]=1.;state,_=step(env,state,action)
                action=np.zeros(11);action[10]=-1.;state,_=step(env,state,action)
                if get(state,"robot","grasp_active")>.5:
                    print("CUBE HIT joint",ji+1,"delta",dq,"xy",[dx,dy],"q",q.tolist(),summary(state,target));env.close();return
        print("cube refine miss joint",ji+1,"delta",dq,flush=True);env.close()


def scan(seed, target, posture, rotation, moving_close):
    """Grid base around an object; open, then exact close pulse at each point."""
    env = make_env(); state, _ = env.reset(seed=seed)
    tx, ty = get(state, target, "pose_x"), get(state, target, "pose_y")
    # Reach requested posture and rotation with gripper definitely open.
    for _ in range(32):
        action = np.zeros(11); action[10] = 1.0
        action[2] = np.clip(rotation-get(state, "robot", "pos_base_rot"), -.2, .2)
        for j, q in enumerate(posture):
            action[3+j] = np.clip(q-get(state, "robot", f"joint_{j+1}"), -.2, .2)
        state, done = step(env, state, action)
        if done: break
    offsets = np.linspace(-.8, .8, 13)
    tested = 0
    for iy, oy in enumerate(offsets):
        row = offsets if iy % 2 == 0 else offsets[::-1]
        for ox in row:
            gx, gy = tx+ox, ty+oy
            while max(abs(gx-get(state,"robot","pos_base_x")),
                      abs(gy-get(state,"robot","pos_base_y"))) > .015:
                action = np.zeros(11); action[10] = 1.0
                action[0] = np.clip(gx-get(state,"robot","pos_base_x"),-.2,.2)
                action[1] = np.clip(gy-get(state,"robot","pos_base_y"),-.2,.2)
                state, done = step(env,state,action)
                if done: break
            # open one full step; close either stationary or while moving slightly.
            action=np.zeros(11); action[10]=1.; state,done=step(env,state,action)
            action=np.zeros(11); action[10]=-1.
            if moving_close:
                action[0] = .015
                action[3] = .015
            state,done=step(env,state,action); tested += 1
            if get(state,"robot","grasp_active") > .5 or get(state,target,"grasp_active") > .5:
                print("HIT",target,"rotation",rotation,"moving",moving_close,
                      "offset",[ox,oy],summary(state,target),
                      "q",[get(state,"robot",f"joint_{j}") for j in range(1,8)])
                # Does holding close retain attachment while base translates?
                before=summary(state,target)
                action=np.zeros(11);action[0]=.2;action[1]=.1;action[10]=-1.
                state,_=step(env,state,action)
                print("attached_move",before,"=>",summary(state,target))
                env.close(); return True
            if done:
                print("ended",tested); env.close(); return False
    print("MISS",target,"rotation",rotation,"moving",moving_close,"tested",tested)
    env.close(); return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("commands","direct","ranges","broad","seeds","route","cube-refine","scan"))
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--target",default="box0")
    parser.add_argument("--posture",choices=("default","low","candidate"),default="default")
    parser.add_argument("--rotation",type=float,default=0.)
    parser.add_argument("--moving-close",action="store_true")
    parser.add_argument("--count",type=int,default=100)
    parser.add_argument("--keep",type=int,default=4)
    args=parser.parse_args()
    postures={
        "default":[0.,-.35,-np.pi,-2.5,0.,-.87,np.pi/2],
        "low":[0.,-1.55,-np.pi,-2.5,0.,-1.9,np.pi/2],
        "candidate":[0.,-1.6743,-3.8876,-1.5737,.1635,-1.7428,.0253],
    }
    if args.mode == "commands": command_probe(args.seed)
    elif args.mode == "direct": direct_probe(args.seed,args.target)
    elif args.mode == "ranges": range_probe(args.seed)
    elif args.mode == "broad": broad_probe(args.seed,args.target)
    elif args.mode == "seeds": seed_sweep(args.seed,args.count)
    elif args.mode == "route": replay_route(args.seed,args.target,args.keep)
    elif args.mode == "cube-refine": cube_refine(args.seed,args.target)
    else: scan(args.seed,args.target,postures[args.posture],args.rotation,args.moving_close)


if __name__ == "__main__":
    main()
