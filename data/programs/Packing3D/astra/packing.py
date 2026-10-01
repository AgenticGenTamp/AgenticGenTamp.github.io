"""Small raster-based greedy packing planner for variable sets of convex parts."""
import numpy as np
from scipy.signal import fftconvolve

def _plan_parts(state, objects, rack, diagonal_penalty=20, reverse=False, alternate=False, cubepad=0):
    spacing=.002
    clearance=.011 if cubepad else .013
    halfx=state.get(rack,'half_extent_x')-clearance
    halfy=state.get(rack,'half_extent_y')-clearance
    width=int(2*halfx/spacing);height=int(2*halfy/spacing)
    free=np.ones((height,width))
    result={}
    descriptions=[]
    for obj in objects:
        if obj.type.name=='Kinematic3DTriangle':
            a=state.get(obj,'side_a');b=state.get(obj,'side_b')
            if state.get(obj,'triangle_type')>.5:
                poly=np.array([[0,0],[a,0],[0,b]])
            else:
                altitude=np.sqrt(max(b*b-a*a/4,1e-8))
                poly=np.array([[-a/2,-altitude/3],[a/2,-altitude/3],[0,2*altitude/3]])
            area=a*b/2
        else:
            a=state.get(obj,'half_extent_x');b=state.get(obj,'half_extent_y')
            poly=np.array([[-a,-b],[a,-b],[a,b],[-a,b]])
            area=4*a*b
        descriptions.append((area,obj,poly))
    descriptions.sort(key=lambda x:-x[0],reverse=reverse)
    for part_index,(area,obj,poly) in enumerate(descriptions):
        best=None
        for angle in [0.,np.pi,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4]:
            c,s=np.cos(angle),np.sin(angle)
            vertices=poly@np.array([[c,s],[-s,c]])
            margin=.001 if cubepad else .002
            low=vertices.min(axis=0)-margin
            high=vertices.max(axis=0)+margin
            w,h=np.ceil((high-low)/spacing).astype(int)
            if w>width or h>height:continue
            yy,xx=np.mgrid[:h,:w]
            points=np.stack([xx*spacing+spacing/2+low[0],yy*spacing+spacing/2+low[1]],axis=-1)
            mask=np.ones((h,w),dtype=bool)
            for p,q in zip(vertices,np.roll(vertices,-1,axis=0)):
                edge=q-p
                cross=edge[0]*(points[:,:,1]-p[1])-edge[1]*(points[:,:,0]-p[0])
                mask &= cross>=-(.0005 if cubepad else .0015)*np.linalg.norm(edge)
            matches=fftconvolve(free,mask[::-1,::-1],mode='valid')
            gridy,gridx=np.indices(matches.shape)
            original_grip=np.array([.025,.025]) if obj.type.name=='Kinematic3DTriangle' and state.get(obj,'triangle_type')>.5 else np.array([-.005,-.005])
            grips=[original_grip]
            if cubepad and obj.type.name=='Kinematic3DTriangle':
                if state.get(obj,'triangle_type')>.5:
                    grips += [np.array(v) for v in [(.02,.045),(.045,.02),(.035,.035),(.015,.055),(.055,.015)]]
                else:
                    grips += [np.array(v) for v in [(0,.015),(-.015,0),(.015,0),(0,-.012)]]
            chosen=np.full(matches.shape,-1,dtype=int)
            for gi,original in enumerate(grips):
                grip=original@np.array([[c,s],[-s,c]])
                gx=-halfx+gridx*spacing-low[0]+grip[0]
                gy=-halfy+gridy*spacing-low[1]+grip[1]
                okay=(abs(gx)<halfx-.035)&(abs(gy)<halfy-.04)
                chosen[(chosen<0)&okay]=gi
            valid=chosen>=0
            if obj.type.name!='Kinematic3DTriangle':valid &= gridy*spacing>=cubepad
            ys,xs=np.where((matches>mask.sum()-.1)&valid)
            if not len(xs):continue
            scores=(ys+h)*width+(width-xs if alternate and part_index%2 else xs)+.001*abs(angle)
            i=np.argmin(scores);score=scores[i]+(diagonal_penalty*width if abs(angle/(np.pi/2)-round(angle/(np.pi/2)))>.01 else 0)
            if best is None or score<best[0]:best=(score,int(xs[i]),int(ys[i]),angle,low,mask,grips[chosen[ys[i],xs[i]]])
        if best is None:
            continue
        _,x,y,angle,low,mask,grip=best
        h,w=mask.shape
        free[y:y+h,x:x+w][mask]=0
        origin=np.array([-halfx+x*spacing,-halfy+y*spacing])-low
        result[obj.name]=(origin,angle,grip)
    return result


def plan_parts(state, objects, rack):
    best={}
    for diagonal_penalty,reverse,alternate in [(20,False,False),(0,False,False),(20,True,False),(0,True,False),(20,False,True),(0,False,True),(20,True,True),(0,True,True)]:
        layout=_plan_parts(state,objects,rack,diagonal_penalty,reverse,alternate)
        if len(layout)>len(best):best=layout
        if len(best)==len(objects):break
    if len(best)<len(objects):
        for pad in [.024,.032,.04]:
            for alternate in [False,True]:
                layout=_plan_parts(state,objects,rack,20,False,alternate,pad)
                if len(layout)>len(best):best=layout
                if len(best)==len(objects):return best
    return best
