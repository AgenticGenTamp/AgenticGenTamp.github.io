import sys; sys.path.insert(0,"/sandbox")
import numpy as np, json
from collections import deque
from scipy.optimize import least_squares
exec(open('/sandbox/cup/an.py').read().split('for s in')[0])
D=json.load(open('/sandbox/cup/rods.json'))
files={'s0':'s0','s1':'s1','s7':'s7','oc3_seed1':'state_seed1_oc3_seed1','oc3_seed2':'state_seed2_oc3_seed2',
 'oc3_seed3':'state_seed3_oc3_seed3','oc3_seed11':'state_seed11_oc3_seed11'}
def blobs(a):
    R,G,B=a[...,0],a[...,1],a[...,2]
    m=((R-B)>=35)&(R>110)&(R<200); m[:150]=False
    ys,xs=np.nonzero(m); S=set(zip(ys.tolist(),xs.tolist())); lab={}; c=0; res=[]
    for p in list(S):
        if p in lab: continue
        q=deque([p]); lab[p]=c; comp=[p]
        while q:
            y,x=q.popleft()
            for dy in range(-3,4):
                for dx in range(-3,4):
                    n=(y+dy,x+dx)
                    if n in S and n not in lab: lab[n]=c; q.append(n); comp.append(n)
        if len(comp)>90 and len(comp)<400:
            arr=np.array(comp); res.append((arr[:,0].mean(),arr[:,1].mean(),len(comp)))
        c+=1
    return res
obs=[]
for k,f in files.items():
    a=load('/sandbox/cup/%s.ppm'%f).astype(int)
    bl=blobs(a); rods=D[k]['rods']
    if len(bl)!=len(rods):
        print('MISMATCH',k,[(round(b[0]),round(b[1]),b[2]) for b in bl],len(rods)); continue
    # match: sort blobs by pixel x asc ; rods by y desc (since +y -> left)
    bl2=sorted(bl,key=lambda t:t[1]); rd=sorted(rods,key=lambda r:-r[1])
    for b,r in zip(bl2,rd): obs.append((r[0],r[1],0.03,b[1],b[0],k))
print('n obs',len(obs))
P=np.array([[o[0],o[1],o[2]] for o in obs]); UV=np.array([[o[3],o[4]] for o in obs])
def rod(rv):
    t=np.linalg.norm(rv)
    if t<1e-9: return np.eye(3)
    k=rv/t; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
def proj(p,pts):
    C=p[:3]; Rm=rod(p[3:6]); f=p[6]
    pc=(Rm@(pts-C).T).T
    return np.stack([320+f*pc[:,0]/pc[:,2], 240+f*pc[:,1]/pc[:,2]],1)
def res(p): return (proj(p,P)-UV).ravel()
# init: camera at (-2,0,2) looking toward +x downward. cam axes: z_cam=view dir, x_cam=right, y_cam=down
def look(C,tgt):
    z=(tgt-C); z/=np.linalg.norm(z)
    up=np.array([0,0,1.0]); x=np.cross(z,up); x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.stack([x,y,z])
best=None
for cx in [-2.5,-1.5,-0.8]:
    for cz in [1.0,1.8,2.6]:
        C=np.array([cx,0.0,cz]); Rm=look(C,np.array([1.0,0,0.2]))
        # rodrigues from matrix
        th=np.arccos(np.clip((np.trace(Rm)-1)/2,-1,1))
        if th<1e-6: rv=np.zeros(3)
        else: rv=th/(2*np.sin(th))*np.array([Rm[2,1]-Rm[1,2],Rm[0,2]-Rm[2,0],Rm[1,0]-Rm[0,1]])
        p0=np.concatenate([C,rv,[579.4]])
        try: s=least_squares(res,p0,method='lm',max_nfev=20000)
        except Exception as e: continue
        if best is None or s.cost<best.cost: best=s
s=best
print('cost',s.cost,'rms px',np.sqrt(np.mean(s.fun**2)))
p=s.x; print('camera pos',p[:3],'f',p[6])
np.save('/sandbox/cup/cam.npy',p)
r=(proj(p,P)-UV)
for i,o in enumerate(obs): print('  %s w(%.2f,%.2f) uv(%.1f,%.1f) resid(%.1f,%.1f)'%(o[5],o[0],o[1],o[3],o[4],r[i,0],r[i,1]))

print('=== VALIDATION: predict cupboard bay centers at x=2.0, z=0 ===')
for yy in [0.25,0.15,0.05,-0.05,-0.15,-0.25]:
    q=proj(p,np.array([[2.0,yy,0.0]]))[0]; print('  y=%+.2f -> u=%.1f v=%.1f'%(yy,q[0],q[1]))
print('=== solve z for given pixel row v at x=xf, y=0 ===')
def solve_z(v,xf,y=0.0):
    from scipy.optimize import brentq
    g=lambda z: proj(p,np.array([[xf,y,z]]))[0][1]-v
    return brentq(g,-0.5,3.0)
for xf in [1.85,1.9,1.95,2.0,2.05,2.1,2.15]:
    print('  xf=%.2f: v152->z=%.3f v150->%.3f v137->%.3f v133->%.3f v129->%.3f v111->%.3f v87->%.3f v67->%.3f v47->%.3f'%(
        xf,solve_z(152,xf),solve_z(150,xf),solve_z(137,xf),solve_z(133,xf),solve_z(129,xf),solve_z(111,xf),solve_z(87,xf),solve_z(67,xf),solve_z(47,xf)))
