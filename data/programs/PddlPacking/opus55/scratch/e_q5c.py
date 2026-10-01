import sys; sys.path.insert(0,"scratch"); from h import *
o,info=env.reset(seed=0); o,_=goto(o,BASE,rstate(o)[3:10]); b=blocks(o)[1]; p=bpose(o,b)
y=best_yaw(o,BASE,[p[0],p[1],0.8],byaw(o,b))
d=np.array([np.cos(y),np.sin(y)])*0.12   # along finger axis
P1=list(p[:2]-d)+[0.80]; P2=list(p[:2]+d)+[0.80]
o,r1=move_tool(o,BASE,P1[:2]+[0.95],y); o,r2=move_tool(o,BASE,P1,y); print("at P1",r1,r2,tool(o)[0].round(3))
s=rstate(o); qn,e=ikp(BASE,P2,y,s[3:10]); print("ik",e)
a=np.zeros(11,np.float32); a[3:10]=qn-s[3:10]; print("maxdelta",np.abs(a).max().round(3))
o2=step(o,a); print("one-step jump through block: tool now",tool(o2)[0].round(3), "block",bpose(o2,b)[:3].round(3))
o3,r=move_tool(o,BASE,P1[:2]+[0.95],y); o3,r=move_tool(o3,BASE,P2[:2]+[0.95],y); o3,r=move_tool(o3,BASE,P2,y); print("P2 via high path rej",r,tool(o3)[0].round(3))
