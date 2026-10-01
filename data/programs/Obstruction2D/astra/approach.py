import math
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.types = {t.name: t for t in observation_space.types}

    def reset(self, state, info):
        self.robot = state.get_objects(self.types['crv_robot'])[0]
        self.target = state.get_objects(self.types['target_block'])[0]
        self.surface = state.get_objects(self.types['target_surface'])[0]
        self.phase = 'select'
        self.item = None
        self.phase_steps = 0
        self.forced = None

    def rect(self, s, o):
        return tuple(s.get(o,k) for k in ('x','y','width','height'))

    def action(self, s, x, y, vac, arm=.2):
        r=self.robot
        angle=(-math.pi/2-s.get(r,'theta')+math.pi)%(2*math.pi)-math.pi
        return np.array([np.clip(x-s.get(r,'x'),-.05,.05),
                         np.clip(y-s.get(r,'y'),-.05,.05),
                         np.clip(angle,-.196,.196),
                         np.clip(arm-s.get(r,'arm_joint'),-.1,.1),vac],dtype=np.float32)

    def enter(self, phase):
        self.phase=phase
        self.phase_steps=0

    def get_action(self,s):
        r=self.robot
        rx,ry=s.get(r,'x'),s.get(r,'y')
        objects=[o for o in s.get_objects(self.types['rectangle']) if not s.get(o,'static')]
        sx,sy,sw,sh=self.rect(s,self.surface)
        tx,ty,tw,th=self.rect(s,self.target)
        self.phase_steps+=1
        if self.phase=='select':
            self.goalx=min(sx+(sw-tw)/2,max(sx+.001,1.49-tw/2))
            obstacles=[]
            for o in objects:
                if o==self.target:continue
                x,y,w,h=self.rect(s,o)
                if x+w>self.goalx-.004 and x<self.goalx+tw+.004 and y<sy+sh+th+.12:
                    obstacles.append(o)
            if obstacles or self.forced is not None:
                self.item=self.forced if self.forced is not None else max(obstacles,key=lambda o:s.get(o,'y')+s.get(o,'height'))
                self.forced=None
                iw=s.get(self.item,'width')
                intervals=[(s.get(o,'x')-.012,s.get(o,'x')+s.get(o,'width')+.012) for o in objects if o!=self.item]
                intervals.append((self.goalx-.05,self.goalx+tw+.05))
                candidates=[.11-iw/2,1.49-iw/2]+[b+.005 for a,b in intervals]+[a-iw-.005 for a,b in intervals]
                candidates=[x for x in candidates if x>=.005 and x+iw<=1.595 and .105<=x+iw/2<=1.495 and all(x+iw<=a or x>=b for a,b in intervals)]
                self.destx=min(candidates,key=lambda x:abs(x-s.get(self.item,'x'))) if candidates else max(.005,.11-iw/2)
                self.desty=.101
                if not candidates:
                    self.desty=max([.101]+[s.get(o,'y')+s.get(o,'height')+.01 for o in objects if o!=self.item and s.get(o,'x')<self.destx+iw and s.get(o,'x')+s.get(o,'width')>self.destx])
            else:
                self.item=self.target
                self.destx=self.goalx
                self.desty=sy+sh+.002
            ix,iy,iw,ih=self.rect(s,self.item)
            lo=max(-.034,.102-ix,.102-self.destx)
            hi=min(iw+.034,1.498-ix,1.498-self.destx)
            grasp_candidates=[float(np.clip(iw/2,lo,hi)),lo,hi]
            for o in objects:
                if o!=self.item:
                    ox,oy,ow,oh=self.rect(s,o)
                    grasp_candidates.extend([ox-.036-ix,ox+ow+.036-ix])
            grasp_candidates=[v for v in grasp_candidates if lo-1e-7<=v<=hi+1e-7]
            if not grasp_candidates:
                grasp_candidates=[hi]
            def blockers(offset):
                px=ix+offset
                py=iy+ih+.2101
                bad=[]
                for o in objects:
                    if o==self.item:continue
                    ox,oy,ow,oh=self.rect(s,o)
                    gripper=(ox<px+.0355 and ox+ow>px-.0355 and oy+oh>iy+ih+.00005 and oy<py)
                    nx=min(max(px,ox),ox+ow)
                    ny=min(max(py,oy),oy+oh)
                    base=(px-nx)**2+(py-ny)**2<.1005**2
                    if gripper or base:bad.append(o)
                return bad
            offset=min(grasp_candidates,key=lambda v:(len(blockers(v)),abs(v-iw/2)))
            bad=blockers(offset)
            if bad:
                taller=[o for o in bad if s.get(o,'y')+s.get(o,'height')>iy+ih+.001]
                if taller:
                    self.forced=max(taller,key=lambda o:s.get(o,'y')+s.get(o,'height'))
                    return self.action(s,rx,ry,0)
            self.pickx=ix+offset
            self.initial_y=iy
            approach_left,approach_right=min(rx,self.pickx)-.11,max(rx,self.pickx)+.11
            tops=[s.get(o,'y')+s.get(o,'height') for o in objects if s.get(o,'x')<approach_right and s.get(o,'x')+s.get(o,'width')>approach_left]
            self.approach_y=min(.895,max([iy+ih]+tops)+.23)
            # Include the base as well as the payload for off-center grasps.
            left_pad=max(.11,.101-offset)
            right_pad=max(.11,offset+.101-iw)
            carry_left=min(ix,self.destx)-left_pad
            carry_right=max(ix,self.destx)+iw+right_pad
            tops=[s.get(o,'y')+s.get(o,'height') for o in objects if o!=self.item and s.get(o,'x')<carry_right and s.get(o,'x')+s.get(o,'width')>carry_left]
            self.clear_y=min(.895,max([iy+.02,self.desty]+tops)+ih+.225)
            self.enter('above')
        ix,iy,iw,ih=self.rect(s,self.item)
        if self.phase=='above':
            if abs(rx-self.pickx)<.002 and abs(ry-self.approach_y)<.002:
                self.enter('descend')
            else:
                # Raise before crossing any objects.
                return self.action(s,self.pickx if ry>=self.approach_y-.002 else rx,self.approach_y,0)
        if self.phase=='descend':
            if abs(ry-(iy+ih+.2101))<.0003 or self.phase_steps>25:
                self.enter('lift')
                return self.action(s,rx,ry,1)
            else:
                return self.action(s,self.pickx,iy+ih+.2101,0)
        if self.phase=='lift':
            if self.phase_steps>3 and iy<self.initial_y+.01:
                self.enter('above')
                return self.action(s,rx,self.approach_y,0)
            self.offsetx=rx-ix
            self.offsety=ry-iy
            if ry>=self.clear_y-.002:
                self.enter('across')
            else:
                return self.action(s,rx,self.clear_y,1)
        if self.phase=='across':
            goalrx=self.destx+self.offsetx
            if abs(rx-goalrx)<.001:
                self.enter('place')
            else:
                return self.action(s,goalrx,self.clear_y,1)
        if self.phase=='place':
            goaly=self.desty+self.offsety
            if abs(ry-goaly)<.001 or self.phase_steps>35:
                self.enter('release')
            else:
                return self.action(s,self.destx+self.offsetx,goaly,1)
        if self.phase=='release':
            self.enter('select')
            return self.action(s,rx,ry,0)
        return self.action(s,rx,ry,0)
