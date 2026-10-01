"""Kinova Gen3 forward kinematics calibrated from black-box observations."""
import numpy as np

_ORIGINS = np.array([
    [.1199, 0., .15643 + .3948],
    [0., .005375, -.12838],
    [0., -.21038, -.006375],
    [0., .006375, -.21038],
    [0., -.20843, -.006375],
    [0., .00017505, -.10593],
    [0., -.10593, -.00017505],
    [0., 0., -.061525 - .12],
])
_RX = np.array([
    [[1,0,0],[0,-1,0],[0,0,-1]],
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
    [[1,0,0],[0,-1,0],[0,0,-1]],
], dtype=float)

def forward(q, base=(0., 0., 0.)):
    """Return world end-effector position and 3x3 orientation."""
    c,s=np.cos(base[2]),np.sin(base[2])
    rot=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    pos=np.array([base[0],base[1],0.],dtype=float)
    for i in range(8):
        pos += rot @ _ORIGINS[i]
        rot = rot @ _RX[i]
        if i < 7:
            c,s=np.cos(q[i]),np.sin(q[i])
            rot=rot@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return pos,rot
