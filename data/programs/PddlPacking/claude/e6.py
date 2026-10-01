import numpy as np, fk
from env_client import make_env

RF=["base_x","base_y","base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7","gripper_opening","grasp_active"]
def rob(s):
    r=s.get_object_from_name("robot"); return np.array([s.get(r,f) for f in RF])
def blocks(s):
    out={}
    for b in s.get_objects(s.get_object_from_name("block0").type):
        out[b.name]=np.array([s.get(b,f) for f in ["pose_x","pose_y","pose_z","pose_qx","pose_qy","pose_qz","pose_qw","grasp_active"]])
    return out

def targR(a):
    x=np.array([0,0,-1.]); y=np.array([np.cos(a),np.sin(a),0]); z=np.cross(x,y)
    return np.column_stack([x,y,z])

env=make_env()
obs,info=env.reset(seed=0)
bl=blocks(obs); print({k:np.round(v[:3],3) for k,v in bl.items()})
tb=bl["block0"]
yaw=2*np.arctan2(tb[5],tb[6])
# choose alignment closest to pi/2
cands=[yaw+k*np.pi/2 for k in range(-4,5)]
a=min(cands,key=lambda c: abs(((c-np.pi/2+np.pi)%(2*np.pi))-np.pi))
print("blockyaw",yaw,"a",a)

def goto(env,obs,base_t,q_t,grip=0.0,maxsteps=100):
    for i in range(maxsteps):
        r=rob(obs)
        db=np.array(base_t)-r[:3]; dq=np.array(q_t)-r[3:10]
        # wrap continuous joints 5,7 -> idx 4,6
        for j in [4,6]:
            dq[j]=(dq[j]+np.pi)%(2*np.pi)-np.pi
        d=np.concatenate([db,dq])
        if np.abs(d).max()<1e-4 and grip==0: break
        a=np.zeros(11); a[:10]=np.clip(d,-0.2,0.2); a[10]=grip
        obs,rew,term,trunc,info=env.step(a)
        if np.abs(d).max()<1e-4: break
    return obs

bx,by=-0.45, float(tb[1])-0.26
obs=goto(env,obs,(bx,by,0.0),rob(obs)[3:10])
print("base now",np.round(rob(obs)[:3],3))
# IK target in base frame
for dz in [0.04,0.0,-0.04]:
    tp=np.array([tb[0]-bx, tb[1]-by, tb[2]+dz])
    q,ok=fk.ik(tp,targR(a),rob(obs)[3:10],0.0)
    p,R,_=fk.fk_arm(q,0.0)
    print("dz",dz,"ok",ok,"err",np.round(p-tp,4),"q",np.round(q,3))
tp=np.array([tb[0]-bx, tb[1]-by, tb[2]+0.02])
q,ok=fk.ik(tp,targR(a),rob(obs)[3:10],0.0)
obs=goto(env,obs,(bx,by,0.0),q)
print("after move q",np.round(rob(obs)[3:10],3),"target",np.round(q,3))
obs,r,t,tr,i=env.step(np.array([0,0,0,0,0,0,0,0,0,0,-1.0]))
print("grasp_active",rob(obs)[11], "gripper",rob(obs)[10])
print(np.round(np.array([obs.get(obs.get_object_from_name("robot"),f) for f in ["grasp_tf_x","grasp_tf_y","grasp_tf_z","grasp_tf_qx","grasp_tf_qy","grasp_tf_qz","grasp_tf_qw"]]),4))
env.close()
