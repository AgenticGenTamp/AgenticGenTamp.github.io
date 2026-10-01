import numpy as np
from env_client import make_env

def get(s,n,f): return float(s.get(s.get_object_from_name(n),f))

def run(seed, count, mode):
    e=make_env(); s,_=e.reset(seed=seed,options={'object_count':count})
    cubes=[n for n in s.get_object_names() if n.startswith('cube_')]
    print('start',seed,count,mode,'r',get(s,'robot','pos_base_x'),get(s,'robot','pos_base_y'),'c',[(n,get(s,n,'x'),get(s,n,'y')) for n in cubes])
    total=0
    for t in range(220):
        a=np.zeros(11,np.float32)
        if mode=='down': a[1]=-.1
        elif mode=='align_down':
            target=sum(get(s,n,'x') for n in cubes)/len(cubes)
            a[0]=np.clip((target-get(s,'robot','pos_base_x'))*.7,-.1,.1)
            a[1]=-.1
        elif mode=='right': a[0]=.1
        elif mode=='wiper_slow':
            wx=get(s,'wiper_0','x')
            if t < 25: a[0]=np.clip((wx-get(s,'robot','pos_base_x'))*.5,-.025,.025)
            else: a[1]=-.01
        elif mode=='wiper_fast':
            # The chassis collides with the wide wiper even though it ghosts through cubes.
            a[1]=-.1
        elif mode=='normal_ram':
            # Push perpendicular to the initial blade yaw, aiming through its COM.
            d=np.array([np.cos(-.813),np.sin(-.813)])
            w=np.array([get(s,'wiper_0','x'),get(s,'wiper_0','y')])
            if t < 18:
                target=np.array([1.5577,.8918])-0.52*d
                a[:2]=np.clip((target-np.array([get(s,'robot','pos_base_x'),get(s,'robot','pos_base_y')]))*.8,-.1,.1)
            else: a[:2]=.1*d
        s,r,term,trunc,info=e.step(a); total+=r
        if r != -.01 or t%40==39:
            print(t,round(r,3),round(total,3),'rxy',round(get(s,'robot','pos_base_x'),2),round(get(s,'robot','pos_base_y'),2),'w',tuple(round(get(s,'wiper_0',f),2) for f in ('x','y','z')),'c',[(round(get(s,n,'x'),2),round(get(s,n,'y'),2)) for n in cubes],term,trunc)
        if term or trunc: break
    e.close()

if __name__=='__main__':
    run(0,1,'normal_ram')
