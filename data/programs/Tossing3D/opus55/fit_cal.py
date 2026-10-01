import numpy as np, kin, glob
from scipy.optimize import least_squares
L=np.concatenate([np.load(f)['log'] for f in glob.glob('pick_log_*.npz')])
# held samples: cube high and robot quasi static (compare consecutive)
idx=[i for i in range(1,len(L)) if L[i,12]>0.08 and L[i,13]>0.5 and np.abs(L[i,3:10]-L[i-1,3:10]).max()<0.003]
print('samples',len(idx),'of',len(L))
D=L[idx]
def res(p,D=D):
    m=p[:3]; my=p[3]; t=p[4:7]
    return np.concatenate([kin.fk(d[0],d[1],d[2],d[3:10],mount=m,mount_yaw=my,tool=t)[0]-d[10:13] for d in D])
p0=np.r_[kin.MOUNT,kin.MOUNT_YAW,kin.TOOL]
r0=res(p0).reshape(-1,3); print('init rms',np.sqrt((r0**2).sum(1).mean()).round(4), 'mean',r0.mean(0).round(4))
s=least_squares(res,p0)
r=s.fun.reshape(-1,3); print('fit',s.x.round(4)); print('fit rms',np.sqrt((r**2).sum(1).mean()).round(4),'max',np.sqrt((r**2).sum(1)).max().round(4))
# cross-validation: fit on seeds 1,2 test on 3
def load(fs):
    L=np.concatenate([np.load(f)['log'] for f in fs])
    idx=[i for i in range(1,len(L)) if L[i,12]>0.08 and L[i,13]>0.5 and np.abs(L[i,3:10]-L[i-1,3:10]).max()<0.003]
    return L[idx]
Dtr=load(['pick_log_1.npz','pick_log_2.npz']); Dte=load(['pick_log_3.npz'])
s2=least_squares(lambda p: res(p,Dtr),p0)
rt=res(s2.x,Dte).reshape(-1,3); print('CV fit',s2.x.round(4),'test rms',np.sqrt((rt**2).sum(1).mean()).round(4),'max',np.sqrt((rt**2).sum(1)).max().round(4), 'n',len(Dtr),len(Dte))
# constrained: tool x,y = 0, mount yaw 0, mount y=0
def res2(p,D=D): return res(np.r_[p[0],0,p[1],0,0,0,p[2]],D)
s3=least_squares(res2,[0.12,0.4,0.13]); r=s3.fun.reshape(-1,3)
print('constrained fit (mx,mz,tz)',s3.x.round(4),'rms',np.sqrt((r**2).sum(1).mean()).round(4),'max',np.sqrt((r**2).sum(1)).max().round(4))
p=np.r_[0.1207,0,0.3947,0,0,0,0.145]
r=res(p).reshape(-1,3)
tilt=np.array([kin.fk(d[0],d[1],d[2],d[3:10],mount=p[:3],mount_yaw=0,tool=p[4:7])[1][2,2] for d in D])
print('corr resid z vs tool-z-down component',np.corrcoef(r[:,2],tilt)[0,1].round(3),'resid mean',r.mean(0).round(4),'std',r.std(0).round(4))
for f in ['pick_log_1.npz','pick_log_2.npz','pick_log_3.npz']:
    rr=res(p,load([f])).reshape(-1,3); print(f,'mean',rr.mean(0).round(4),'std',rr.std(0).round(4))
