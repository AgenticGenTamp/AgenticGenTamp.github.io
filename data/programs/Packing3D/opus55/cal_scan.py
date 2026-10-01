from cal_trial import *
import sys
def show(lbl,t): print(f"{lbl} reached={t['reached']} tcp_rel={np.round(t['tcp'],3)} ga={t['ga']} pga={t['pga']}",flush=True)
for dz in [0.085,0.09,0.095,0.035,0.038]: show(f'dz={dz}',trial(0,'part0',0,0,dz))
for dx in [0.02,0.04,0.05,0.06,0.08]: show(f'dx={dx}',trial(0,'part0',dx,0,0.05))
for dx in [-0.04,-0.06,-0.08]: show(f'dx={dx}',trial(0,'part0',dx,0,0.05))
for dy in [0.04,0.06,-0.04,-0.06]: show(f'dy={dy}',trial(0,'part0',0,dy,0.05))
for dxy in [0.03,0.04,0.05]: show(f'diag={dxy}',trial(0,'part0',dxy,dxy,0.05))
