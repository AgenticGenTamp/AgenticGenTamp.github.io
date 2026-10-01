from env_client import make_env
import numpy as np

Q=np.array([0.,1.3,np.pi,-1.7,0.,1.,0.])
def get(s,n,fs):
 o=s.get_object_from_name(n); return np.array([float(s.get(o,f)) for f in fs])
def robot(s): return get(s,"robot",["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"])
def xyz(s,n): return get(s,n,["x","y","z"])
def action(s,b=None,q=None,g=0):
 r=robot(s); a=np.zeros(11,np.float32)
 if b is not None:a[:3]=np.clip(.8*(np.asarray(b)-r[:3]),-.1,.1)
 if q is not None:a[3:10]=np.clip(.7*(np.asarray(q)-r[3:10]),-.1,.1)
 a[10]=g; return a
def steps(e,s,n,b=None,q=None,g=0):
 for _ in range(n):s,*_=e.step(action(s,b,q,g))
 return s

def trial(bx):
 e=make_env(); s,info=e.reset(seed=0,options={"object_count":4}); home=robot(s)[3:10]; name="cube4"; y=float(xyz(s,name)[1])
 # outside route to left, configure only once safely parked
 s=steps(e,s,45,[1,.8,np.pi],home,0); s=steps(e,s,55,[-1,.8,np.pi],home,0)
 s=steps(e,s,45,[-1,.8,0],home,0); s=steps(e,s,45,[-1,y,0],home,0); s=steps(e,s,130,[-1,y,0],Q,0)
 p0=xyz(s,name).copy(); s=steps(e,s,30,[bx,y,0],Q,0); pre=xyz(s,name).copy()
 s=steps(e,s,15,[bx,y,0],Q,1); closed=xyz(s,name).copy()
 # pull away horizontally, then perturb shoulder upward in both likely sense
 s=steps(e,s,15,[bx-.12,y,0],Q,1); pulled=xyz(s,name).copy()
 qup=Q.copy(); qup[1]-=.35; qup[3]+=.35
 s=steps(e,s,30,[bx-.12,y,0],qup,1); lifted=xyz(s,name).copy()
 print("TRIAL",bx,"base",np.round(robot(s)[:3],3),"grip",robot(s)[-1],"p0",np.round(p0,4),"pre",np.round(pre,4),"closed",np.round(closed,4),"pulled",np.round(pulled,4),"lifted",np.round(lifted,4),"d_pull",np.round(pulled-closed,4),"d_lift",np.round(lifted-pulled,4))
 e.close()

if __name__=="__main__":
 for x in [-.94,-.925,-.91,-.895]:trial(x)
