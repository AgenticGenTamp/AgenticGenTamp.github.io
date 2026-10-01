import json,glob,math
import numpy as np
pts=[]
for f in glob.glob("probe_rew_traj_s11_*.json"):
    pts+=json.load(open(f))
P=np.array(pts); print("traj pts",len(P))
cx,cy=0.851,0.0
covered=0; tot=0; uncov=[]
for gx in np.arange(cx-0.6,cx+0.6001,0.05):
    for gy in np.arange(cy-0.6,cy+0.6001,0.05):
        if math.hypot(gx-cx,gy-cy)>0.6: continue
        tot+=1
        d=np.min(np.hypot(P[:,0]-gx,P[:,1]-gy))
        if d<0.05: covered+=1
        else: uncov.append((round(gx,2),round(gy,2),round(float(d),2)))
print("grid pts within 0.6m:",tot,"covered(<5cm of a visited chair pos):",covered)
print("uncovered sample:",uncov[:15])
