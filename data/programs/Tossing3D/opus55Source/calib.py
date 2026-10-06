import numpy as np, sys, json
import throw2
qw=[0,-1.0,3.1416,0.6,0,0.6,1.5708]; w=[0,1,0,-1,0,-1,0]
def land(bt, traj, zc=0.1):
    for i in range(1,len(traj)):
        if traj[i-1][2]>zc and traj[i][2]<=zc and traj[i-1][9]<0:
            # ballistic interp from previous step
            p=traj[i-1][:3]; v=traj[i-1][7:10]
            # solve p_z + v_z t - 4.905 t^2 = zc
            a=-4.905; b=v[2]; c=p[2]-zc
            t=(-b-np.sqrt(b*b-4*a*c))/(2*a)
            x=p[:2]+v[:2]*t; vz=v[2]-9.81*t
            d=x-bt[:2]; ch,sh=np.cos(bt[2]),np.sin(bt[2])
            return np.array([ch*d[0]+sh*d[1], -sh*d[0]+ch*d[1]]), np.degrees(np.arctan2(-vz, np.linalg.norm(v[:2])))
    return None, None
if __name__=='__main__':
    for V in json.loads(sys.argv[1]):
        for seed in json.loads(sys.argv[2]):
            bt,c0,traj=throw2.run(seed,qw,w,V,int(sys.argv[3]),5,wait=20,verbose=False)
            fl,ang=land(bt,traj)
            print('V',V,'seed',seed,'land fwd/lat',None if fl is None else fl.round(3),'angle',ang)
