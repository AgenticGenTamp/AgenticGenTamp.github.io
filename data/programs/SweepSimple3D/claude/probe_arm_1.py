from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=1)
R=lambda f: float(obs.get(obs.get_object_from_name("robot"),f))
def C(f):
    try: return float(obs.get(obs.get_object_from_name("cube_0"),f))
    except Exception: return float('nan')
names=[o.name for o in obs]
print("objs",names)
print("init robot", [round(R(f),3) for f in ["pos_base_x","pos_base_y","pos_base_rot"]],
      "arm",[round(R("pos_arm_joint%d"%i),3) for i in range(1,8)],"grip",round(R("pos_gripper"),3))
print("cube",round(C("x"),3),round(C("y"),3),round(C("z"),3))
def step(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.asarray(a,dtype=np.float32))
    return rew
# frame test at reset yaw
a=np.zeros(11); a[0]=0.1
for i in range(15): step(a)
print("after +x15 @yaw%.3f:"%R("pos_base_rot"), round(R("pos_base_x"),3), round(R("pos_base_y"),3))
a=np.zeros(11); a[1]=0.1
for i in range(15): step(a)
print("after +y15:", round(R("pos_base_x"),3), round(R("pos_base_y"),3), "yaw",round(R("pos_base_rot"),3))
# rotate to yaw 0
for i in range(40):
    a=np.zeros(11); a[2]=np.clip(0.0-R("pos_base_rot"),-0.1,0.1); step(a)
    if abs(R("pos_base_rot"))<1e-3: break
print("yaw now",round(R("pos_base_rot"),4),"pos",round(R("pos_base_x"),3),round(R("pos_base_y"),3))
x0,y0=R("pos_base_x"),R("pos_base_y")
a=np.zeros(11); a[0]=0.1
for i in range(15): step(a)
print("after +x15 @yaw0: d=",round(R("pos_base_x")-x0,3), round(R("pos_base_y")-y0,3))
print("cube now",round(C("x"),3),round(C("y"),3),round(C("z"),3))
env.close()
