import sys; sys.path.insert(0,"scratch"); from h import *
BASE=(-0.6,0.0,0.0)
def setup(seed=0,bi=1):
    o,info=env.reset(seed=seed)
    s=rstate(o); o,_=goto(o,BASE,s[3:10])
    return o
def trial(z,dx=0,dy=0,dyaw=0,seed=0,bi=1,above=0.95):
    o=setup(seed,bi); b=blocks(o)[bi]; p=bpose(o,b); yaw=byaw(o,b)+dyaw
    tgt=[p[0]+dx,p[1]+dy]
    o,rej=move_tool(o,BASE,tgt+[above],yaw)
    if rej: return "rej_above"
    o,rej=move_tool(o,BASE,tgt+[z],yaw)
    tz=tool(o)[0][2]
    o=grip(o,-1)
    s=rstate(o); ga=[bpose(o,bb)[7] for bb in blocks(o)]
    return dict(rej=rej,tz=round(tz,3),gopen=round(s[10],3),ga=s[11],bga=ga, gtf=s[12:15].round(3))
if __name__=="__main__":
    for z in [0.88,0.86,0.84,0.83,0.82,0.81,0.80,0.79,0.785,0.78,0.77,0.76]:
        print(z,trial(z))
