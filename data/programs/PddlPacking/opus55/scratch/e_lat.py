import sys; sys.path.insert(0,"scratch"); from e_z import *
o=setup(); b=blocks(o)[1]; yaw=byaw(o,b)
fy=np.array([np.cos(yaw),np.sin(yaw)]); fz=np.array([-np.sin(yaw),np.cos(yaw)])
def short(r): return r if isinstance(r,str) else (r['rej'],r['tz'],r['ga'],r['gtf'])
for d in [0.02,0.03,0.04,0.05,0.06]:
    print("along finger axis",d,short(trial(0.80,*(fy*d))))
    print("perp finger axis ",d,short(trial(0.80,*(fz*d))))
for dy in [np.deg2rad(20),np.deg2rad(45),np.pi/2]:
    for z in [0.84,0.80]:
        print("yaw off",round(np.rad2deg(dy)),z,short(trial(z,dyaw=dy)))
