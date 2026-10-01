import sys; sys.path.insert(0,"scratch"); from e_z import *
def single(z0,z1,g):
    o=setup(); b=blocks(o)[1]; p=bpose(o,b); yaw=byaw(o,b)
    o,_=move_tool(o,BASE,[p[0],p[1],0.95],yaw); o,_=move_tool(o,BASE,[p[0],p[1],z0],yaw)
    s=rstate(o); q1,err=ikp(BASE,[p[0],p[1],z1],yaw,s[3:10])
    a=np.zeros(11,np.float32); a[3:10]=q1-s[3:10]; a[10]=g
    print(" maxdelta",np.abs(a[3:10]).max().round(3))
    o=step(o,a); s2=rstate(o)
    print(z0,"->",z1,"grip",g,"tool z",tool(o)[0][2].round(3),"ga",s2[11],"gtf",s2[12:15].round(3),"gopen",s2[10])
    return o
single(0.90,0.80,-1)
single(0.80,0.90,-1)
