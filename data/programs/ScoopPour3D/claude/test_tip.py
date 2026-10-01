"""Tip the yellow bin over its far edge (pivot on counter) into the adjacent green bin."""
from env_client import make_env
import numpy as np, kin, sys
seed=int(sys.argv[1]); oc=int(sys.argv[2]); WX=float(sys.argv[3]); AMAX=float(sys.argv[4]); ZPIV=float(sys.argv[5]); GAP=float(sys.argv[6])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count':oc})
R=obs.get_object_from_name('robot'); GB=obs.get_object_from_name('bin_green_0'); YB=obs.get_object_from_name('bin_yellow_0')
CUBES=[o for o in obs.data if o.name.startswith('cube_')]
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
def gb(): return obs.data[GB][:3].copy()
def yb(): return obs.data[YB][:7].copy()
def cpos(): return np.array([obs.data[c][:3] for c in CUBES])
TOOL=0.12; G=[0.0]; QD=[None]; sc=[0]; RW=[-1.0]; TERM=[False]
def step(bc=(0,0,0)):
    global obs
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(bc,-0.1,0.1); a[10]=G[0]
    if QD[0] is not None: a[3:10]=np.clip(QD[0]-qq(),-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a); sc[0]+=1; RW[0]=rew; TERM[0]=term or trunc
def Rx(t):
    c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def set_target(worldy, zarm, Rt, n=1, bc_gain=0.6):
    """drive base y toward worldy and arm to (fx,0,zarm) with orientation Rt"""
    fx = WX-base()[0]
    qd,err=kin.ik(np.array([fx,0.0,zarm]), Rt, qq(), TOOL); QD[0]=qd
    for k in range(n):
        dy = worldy-base()[1]
        step((0,np.clip(dy*bc_gain,-0.05,0.05),0))
    return err
def base_to(bx,by,bth=0.0,n=40,tol=0.01):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-0.1,0.1))
def arm_to(tgt,Rt=None,n=60,tol=0.008):
    if Rt is None: Rt=kin.rot_down(1.5708)
    qd,err=kin.ik(np.array(tgt), Rt, qq(), TOOL); QD[0]=qd
    for k in range(n):
        step()
        if np.abs(qd-qq()).max()<tol: break
    return err,np.abs(qd-qq()).max()
g0=gb().copy()
# 1) push yellow bin +y until adjacent to green
base_to(-0.16, -0.36, 0.0)
fx=WX-base()[0]
G[0]=1.0
arm_to((fx,0.0,0.34)); arm_to((fx,0.0,0.21),n=40)
tgt_y = g0[1]-0.30-GAP
for i in range(60):
    y=yb()[1]
    if y>tgt_y-0.004: break
    step((0,min(0.04,max(0.01,(tgt_y-y)*0.3)),0))
print('yellow',np.round(yb()[:3],3),'green',np.round(gb(),3),'steps',sc[0])
# 2) grasp -y wall
G[0]=0.0
for i in range(6): step()
ybpos=yb()
arm_to((fx,0.0,0.20),n=30)
G[0]=1.0
for i in range(10): step()
print('grasped, yellow',np.round(yb()[:3],3),'q',np.round(yb()[3:],2),'steps',sc[0])
# 3) arc
Py = yb()[1]+0.15
v0 = np.array([-(0.155+0.15), 0.20-ZPIV])
print('pivot y',round(Py,3))
N=18
for i in range(1,N+1):
    a = -AMAX*i/N
    c,s=np.cos(a),np.sin(a)
    yv = c*v0[0]-s*v0[1]; zv = s*v0[0]+c*v0[1]
    wy = Py+yv; za = ZPIV+zv
    e=set_target(wy, za, Rx(a)@kin.rot_down(1.5708), n=8)
    cc=cpos()
    print(' a',round(a,2),'ik',round(e,3),'wy',round(wy,3),'za',round(za,3),'yb',np.round(yb()[:3],3),'cz',round(cc[:,2].mean(),3),'cy',round(cc[:,1].mean(),3),'green',np.round(gb()[:2],3),'T',TERM[0])
    if TERM[0]: break
for i in range(20): step()
c=cpos(); g=gb()
inb=((np.abs(c[:,0]-g[0])<0.20)&(np.abs(c[:,1]-g[1])<0.13)&(c[:,2]>0.44)&(c[:,2]<0.56)).sum()
print('FINAL rew',RW[0],'term',TERM[0],'steps',sc[0],'green',np.round(g,3),'inbin',inb)
print('cube mean',np.round(c.mean(0),3),'std',np.round(c.std(0),3),'zmin',round(c[:,2].min(),3))
env.close()
