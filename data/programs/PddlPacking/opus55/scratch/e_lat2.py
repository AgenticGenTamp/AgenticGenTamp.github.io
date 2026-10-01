import sys; sys.path.insert(0,"scratch"); from e_z import *
o=setup(); b=blocks(o)[1]; yaw=byaw(o,b)
fy=np.array([np.cos(yaw),np.sin(yaw)]); fz=np.array([-np.sin(yaw),np.cos(yaw)])
def short(r): return r if isinstance(r,str) else (r['rej'],r['tz'],r['ga'],r['gtf'])
for d in [0.005,0.01,0.015]:
    print("along finger axis",d,short(trial(0.80,*(fy*d))))
for d in [0.025,0.03,0.035]:
    print("perp finger axis ",d,short(trial(0.80,*(fz*d))))
for dg in [30,35,40]:
    print("yaw off",dg,short(trial(0.80,dyaw=np.deg2rad(dg))))
for dg in [20,-30]:
    o=setup(); b=blocks(o)[1]; p=bpose(o,b); y=byaw(o,b)+np.deg2rad(dg)
    o,_=move_tool(o,BASE,[p[0],p[1],0.95],y); o,_=move_tool(o,BASE,[p[0],p[1],0.80],y); o=grip(o,-1)
    s=rstate(o); print(dg,"gtf",s[12:19].round(3),"block yaw",round(byaw(o,b),3),"orig",round(byaw(o,b)-0,3), "gopen",s[10])
    o,_=move_tool(o,BASE,[p[0],p[1],0.95],y); print(" after lift block",bpose(o,b).round(3),round(byaw(o,b),3))
