import sys; sys.path.insert(0,"scratch"); from e_z import *
def grasp_lift(seed=0,bi=1,z=0.80,dyaw=0.0,lift=0.95):
    o=setup(seed); b=blocks(o)[bi]; p=bpose(o,b); yaw=byaw(o,b)+dyaw
    o,_=move_tool(o,BASE,[p[0],p[1],0.95],yaw); o,_=move_tool(o,BASE,[p[0],p[1],z],yaw); o=grip(o,-1)
    o,_=move_tool(o,BASE,[p[0],p[1],lift],yaw)
    return o,b,yaw
if __name__=="__main__":
  for (tx,ty,tz,tyaw) in [(0,0,0.95,None),(0.05,-0.03,1.1,None),(0.0,0.0,0.95,0.3)]:
    o,b,yaw=grasp_lift()
    y=yaw if tyaw is None else yaw+tyaw
    o,rej=move_tool(o,BASE,[tx,ty,tz],y); print("move rej",rej)
    print("held block",bpose(o,b).round(3),round(byaw(o,b),3),"tool",tool(o)[0].round(3), "gopen",rstate(o)[10])
    o=grip(o,1)
    print(" after open",bpose(o,b).round(3),round(byaw(o,b),3),"gopen",rstate(o)[10],"ga",rstate(o)[11], LAST['r'],LAST['te'])
    o=grip(o,0)
    print(" next",bpose(o,b).round(3),"gopen",rstate(o)[10])
