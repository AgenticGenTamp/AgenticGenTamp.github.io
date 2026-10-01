import sys; sys.path.insert(0,"scratch"); from e_z import *
for d in [0.03,0.035,0.04,-0.03]:
    o=setup(); b=blocks(o)[1]; p=bpose(o,b); yaw=byaw(o,b)
    fz=np.array([-np.sin(yaw),np.cos(yaw)])
    o,r1=move_tool(o,BASE,[p[0],p[1],0.95],yaw)
    t=list(p[:2]+fz*d)
    o,r2=move_tool(o,BASE,t+[0.95],yaw)
    o,r3=move_tool(o,BASE,t+[0.80],yaw); o=grip(o,-1); s=rstate(o)
    print(d,r1,r2,r3,tool(o)[0].round(3),s[11],s[12:15].round(3))
