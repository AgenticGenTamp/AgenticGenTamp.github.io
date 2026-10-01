import json,glob
import numpy as np
pts=[]
for f in glob.glob("probe_rew_traj_s11_*.json"): pts+=json.load(open(f))
P=np.array(pts)
for t in [(0.94,-0.06),(0.851,0.0),(0.95,-0.15)]:
    d=np.hypot(P[:,0]-t[0],P[:,1]-t[1])
    print("target",t,"min chair dist over all pushes:",round(float(d.min()),3),"num within 5cm:",int((d<0.05).sum()),"within 10cm:",int((d<0.10).sum()))
