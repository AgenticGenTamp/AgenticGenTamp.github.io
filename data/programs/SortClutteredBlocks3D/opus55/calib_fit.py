import json, numpy as np
from scipy.optimize import least_squares
from kin import fk_arm
def Rz(t): return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
H=[r for r in json.load(open('calib_hold.json')) if r['c'][2]>0.45]
T=[r for r in json.load(open('calib_touch.json'))]
fkH=[fk_arm(np.array(r['q']),0) for r in H]; fkT=[fk_arm(np.array(r['q']),0) for r in T]
def resH(x,yaw=True):
    M=x[:3]; v=x[3:6]; dl=x[6] if yaw else 0
    out=[]
    for r,(t,R,_,_) in zip(H,fkH):
        b=np.array(r['base']); out.append(b*[1,1,0]+Rz(b[2]+dl)@(M+t+R@v)-np.array(r['c']))
    return np.concatenate(out)
for yaw in [False,True]:
    s=least_squares(lambda x:resH(x,yaw),np.r_[0.12,0,0.3,0,0,-0.2,0])
    e=resH(s.x,yaw).reshape(-1,3)
    print('HOLD yaw' if yaw else 'HOLD noyaw','M',s.x[:3].round(4),'v',s.x[3:6].round(4),'dyaw',round(s.x[6],4),'rms',np.sqrt((e**2).mean(0)).round(4),'max',np.abs(e).max().round(4),'n',len(H))
# touch fit
Lt=0.21
def resT(x,yaw=True,filt=None):
    mx,my,dl,wx,wy=x; out=[]
    for r,(t,R,_,_) in zip(T,fkT):
        b=np.array(r['base']); d=np.array(r['d'])
        p=b*[1,1,0]+Rz(b[2]+(dl if yaw else 0))@(np.array([mx,my,0])+t+R@[0,0,-Lt])
        w=wx if abs(d[0])>0.5 else wy
        out.append(np.dot(p-np.array(r['c']),d)+w)
    return np.array(out)
s=least_squares(lambda x:resT(x),[0.12,0,0,0.02,0.025])
e=resT(s.x); keep=np.abs(e)<0.004
T=[t for t,k in zip(T,keep) if k]; fkT=[f for f,k in zip(fkT,keep) if k]
for yaw in [False,True]:
    s=least_squares(lambda x:resT(x,yaw),[0.12,0,0,0.02,0.025]); e=resT(s.x,yaw)
    print('TOUCH', 'yaw' if yaw else 'noyaw','mx,my',s.x[:2].round(4),'dyaw',round(s.x[2],4),'wx,wy',s.x[3:].round(4),'rms',np.sqrt((e**2).mean()).round(4),'max',np.abs(e).max().round(4),'n',len(T))
runs=sorted(set(r['seed'] for r in H)); idx=[runs.index(r['seed']) for r in H]
def resH2(x):
    M=x[:3]; out=[]
    for r,(t,R,_,_),k in zip(H,fkH,idx):
        v=x[3+3*k:6+3*k]; b=np.array(r['base']); out.append(b*[1,1,0]+Rz(b[2])@(M+t+R@v)-np.array(r['c']))
    return np.concatenate(out)
s=least_squares(resH2,np.r_[0.12,0,0.3,[0,0,-0.2]*len(runs)]); e=resH2(s.x).reshape(-1,3)
print('HOLD per-run v: M',s.x[:3].round(4),'v',s.x[3:].reshape(-1,3).round(4),'rms',np.sqrt((e**2).mean(0)).round(4))
print('per-sample err norm',np.linalg.norm(e,axis=1).round(4))
# fix M xy from touch, fit mz
def resH3(x):
    M=np.array([0.1198,-0.0006,x[0]]); out=[]
    for r,(t,R,_,_),k in zip(H,fkH,idx):
        v=x[1+3*k:4+3*k]; b=np.array(r['base']); out.append(b*[1,1,0]+Rz(b[2])@(M+t+R@v)-np.array(r['c']))
    return np.concatenate(out)
s=least_squares(resH3,np.r_[0.3,[0,0,-0.2]*len(runs)]); e=resH3(s.x).reshape(-1,3)
print('HOLD Mxy fixed: mz',s.x[0].round(4),'v',s.x[1:].reshape(-1,3).round(4),'rms',np.sqrt((e**2).mean(0)).round(4))
