from cal_trial import *
def s(lbl,t): print(f"{lbl} reached={t['reached']} tcp_rel={np.round(t['tcp'],4)} ga={t['ga']}",flush=True)
for dz in [0.05,0.085]:
  for dx in [0.015,0.019,0.021,0.025]:
    s(f'dz={dz} dx={dx}',trial(0,'part0',dx,0,dz))
for yaw in [0.785,1.571,3.0]:
  for dx,dy in [(0,0),(0.015,0),(0,0.015),(0.025,0)]:
    s(f'yaw={yaw} dx={dx} dy={dy}',trial(0,'part0',dx,dy,0.05,yaw=yaw))
