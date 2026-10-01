from th import *
import sys
dz=float(sys.argv[1]); yaw=float(sys.argv[2])
h=H(1); p=h.pose('target_block'); he=h.he('target_block'); top=p[2]+he[2]
res=[]
for dx in np.arange(-0.06,0.061,0.02):
  row=''
  for dy in np.arange(-0.06,0.061,0.02):
    h.goto([p[0],p[1],top+0.1],yaw)
    ok,_=h.goto([p[0]+dx,p[1]+dy,top+dz],yaw)
    h.grip(-1); g=h.grasped(); row+=('G' if g else '.') if ok else 'x'
    h.grip(1)
    if g:
        h=H(1)
  res.append(row)
print(f'dz={dz} yaw={yaw} he={he.round(3)} (rows dx, cols dy)'); print('\n'.join(res))
