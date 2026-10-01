import sys, time; sys.path.insert(0,'scratch')
from gm_lib import *
grip=float(sys.argv[1]) if len(sys.argv)>1 else 0.0
zs=[float(x) for x in sys.argv[2].split(',')]
dirs=sys.argv[3].split(',') if len(sys.argv)>3 else ["+y","-y","+z","-z"]
yoff=float(sys.argv[4]) if len(sys.argv)>4 else 0.0   # extra yaw offset (deg)
def fresh():
    r=R(27,(-0.47,0.22,0.0))
    if grip<0:
        g=r.grip(-1.0)
    return r
r=fresh(); bc,byaw=block(r.o); yaw=byaw+math.radians(yoff)
c,s=math.cos(yaw),math.sin(yaw)
A={"+y":np.array([c,s,0.]),"+z":np.array([s,-c,0.])}; A["-y"]=-A["+y"]; A["-z"]=-A["+z"]
t0=time.time()
for zt in zs:
    row=[]
    for name in dirs:
        u=A[name]; r=fresh()
        pn=np.array([bc[0],bc[1],zt]); pf=pn+0.16*u
        try: d=r.sweep(pf,pn,yaw)
        except AssertionError as e: d=float('nan'); print("err",e)
        row.append("%s %.4f"%(name,d))
    print("z_t=%.3f "%zt,"  ".join(row),"grip_open",round(rstate(r.o)[10],3),"t",round(time.time()-t0,1),flush=True)
