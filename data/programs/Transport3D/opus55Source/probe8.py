exec(open("/tmp/probe4.py").read().split("cube = np.array")[0])
import sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
s,_ = env.reset(seed=seed)
nsteps=[0]
_orig=env.step
def counted(a):
    nsteps[0]+=1; return _orig(a)
env.step=counted
def goto_base(x,y):
    step_to(base=(-0.3, s.base_pose.y,0)); step_to(base=(-0.3, y, 0)); return step_to(base=(x,y,0))
SAFE = ik  # placeholder
def pick(name, offset=np.zeros(3), zoff=0.03):
    global s
    pos = np.array(s.get_object_pose(name).position)+offset
    step_to(ik(np.array([0.3,0,0.6])+[s.base_pose.x,s.base_pose.y,0]))
    goto_base(pos[0]-0.5, pos[1])
    step_to(ik(pos+[0,0,0.2])); step_to(ik(pos+[0,0,zoff]))
    s,*_ = env.step(np.array([0]*10+[-1],dtype=np.float32))
    P("pick",name,"->",s.grasped_object, "ee", np.round(arm.get_end_effector_pose().position,3))
    step_to(ik(pos+[0,0,0.55]))
def place_table(tgt_xy, hz, held_off=0.0):
    global s
    step_to(ik(np.array([0.3,0,0.65])+[s.base_pose.x,s.base_pose.y,0]))
    goto_base(0.0, 0.0)
    tgt = np.array([tgt_xy[0], tgt_xy[1], 0.4+hz+held_off])
    a=step_to(ik(tgt+[0,0,0.15])); b=step_to(ik(tgt+[0,0,0.004]))
    s,_,term,*_ = env.step(np.array([0]*10+[1],dtype=np.float32))
    P("place", a, b, "held now", s.grasped_object, "term", term)
    step_to(ik(tgt+[0,0,0.3]))
pick("cube0"); g=s.get(s.robot,"grasp_tf_z"); place_table((0.5,0.15), 0.025, held_off=g)  # ee is above object center by tf_z
pick("box0", offset=np.array([-0.095,0,0.1]), zoff=0.0); 
P("box tf", np.round([s.get(s.robot,"grasp_tf_"+k) for k in "xyz"],3))
place_table((0.5-0.095,-0.15), 0.1, held_off=0.0)
P("goal", env.goal_reached(), "steps", nsteps[0], [ (n, np.round(s.get_object_pose(n).position,3)) for n in ["cube0","box0"]])
