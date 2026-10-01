from th import *
import sys
seed=int(sys.argv[1]); name=sys.argv[2]; yaw=float(sys.argv[3]); dx=float(sys.argv[4]); dy=float(sys.argv[5])
h=H(seed); p=h.pose(name); he=h.he(name); top=p[2]+he[2]
h.goto([p[0]+dx,p[1]+dy,top+float(sys.argv[6])+0.02],yaw)
out=[]
for dz in np.arange(float(sys.argv[6]),-0.035,-float(sys.argv[7])):
    ok,_=h.goto([p[0]+dx,p[1]+dy,top+dz],yaw)
    if not ok: out.append(f'{dz:+.3f}:X'); break
    h.grip(-1)
    if h.grasped(): out.append(f'{dz:+.3f}:G tf={np.array([h.obs.get(h.R,f) for f in GF[:3]]).round(3)}'); break
    out.append(f'{dz:+.3f}:.')
print(seed,name,'yaw',yaw,'d',dx,dy,'he',he.round(3),' '.join(out))
