from env_client import make_env
import numpy as np

HIGH=np.array([0.,.90,np.pi,-1.7,0.,1.,np.pi/2])
LOW=np.array([0.,1.30,np.pi,-1.7,0.,1.,np.pi/2])
def get(s,n,fs):
 o=s.get_object_from_name(n); return np.array([float(s.get(o,f)) for f in fs])
def rob(s): return get(s,"robot",["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"])
def xyz(s,n): return get(s,n,["x","y","z"])
def act(s,b=None,q=None,g=0):
 r=rob(s); a=np.zeros(11,np.float32)
 if b is not None:a[:3]=np.clip(.8*(np.asarray(b)-r[:3]),-.1,.1)
 if q is not None:a[3:10]=np.clip(.7*(np.asarray(q)-r[3:10]),-.1,.1)
 a[10]=g; return a
def run(e,s,n,b=None,q=None,g=0):
 for _ in range(n):s,*_=e.step(act(s,b,q,g))
 return s
def trial(offset,open_g=0,close_g=1,low_q2=1.30):
 e=make_env();s,info=e.reset(seed=0,options={"object_count":4}); home=rob(s)[3:10]; target="cube4"; y=float(xyz(s,target)[1]); names=[n for n in s.get_object_names() if n.startswith("cube")]
 # route outside to left with folded arm
 s=run(e,s,45,[1,.8,np.pi],home,open_g);s=run(e,s,55,[-1,.8,np.pi],home,open_g);s=run(e,s,45,[-1,.8,0],home,open_g)
 s=run(e,s,45,[-1,y,0],home,open_g);s=run(e,s,130,[-1,y,0],HIGH,open_g)
 xhigh=-.925+offset; xlow=xhigh
 s=run(e,s,25,[xhigh,y,0],HIGH,open_g); before={n:xyz(s,n).copy() for n in names}
 low=LOW.copy(); low[1]=low_q2
 # Shoulder-only top-down descent; q4 stays fixed to avoid horizontal rake.
 for k in range(60):
  s,*_=e.step(act(s,[xlow,y,0],low,open_g))
 down={n:xyz(s,n).copy() for n in names}
 s=run(e,s,18,[xlow,y,0],low,close_g); closed={n:xyz(s,n).copy() for n in names}
 # Reverse shoulder lift while holding closure.
 maxz={n:closed[n][2] for n in names}
 for k in range(65):
  s,*rest=e.step(act(s,[xhigh,y,0],HIGH,close_g))
  for n in names:maxz[n]=max(maxz[n],xyz(s,n)[2])
 after={n:xyz(s,n).copy() for n in names}
 print("TRIAL",offset,"q2",low_q2,"g",open_g,close_g,"robot",np.round(rob(s),3).tolist(),"down_d",{n:np.round(down[n]-before[n],4).tolist() for n in names},"close_d",{n:np.round(closed[n]-down[n],4).tolist() for n in names},"lift_d",{n:np.round(after[n]-closed[n],4).tolist() for n in names},"maxz",{n:round(maxz[n],4) for n in names})
 e.close()
if __name__=="__main__":
 for off in [-.015,-.010,-.005]:trial(off,1,0,1.24)
