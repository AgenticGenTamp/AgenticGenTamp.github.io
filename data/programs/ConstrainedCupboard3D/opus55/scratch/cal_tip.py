import sys; sys.path.insert(0,'/sandbox/scratch')
from plib import *
p=P(1); S=p.S; S.grip=1.0
goto_slow(S, bt=np.array([0.5,0,0]), steps=100)
print('floor down', probe_dir(p,[1.0,0,0.15],[0,0,-1],0.2,Rdown(0)))
print('floor down2', probe_dir(p,[1.0,0.2,0.15],[0,0,-1],0.2,Rdown(np.pi/2)))
print('floor horiz', probe_dir(p,[1.0,0,0.15],[0,0,-1],0.2,RH))
print(len(S.rew), LOG)
