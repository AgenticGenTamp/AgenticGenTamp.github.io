"""Object-centric feedback controller for Packing3DEnv."""
import numpy as np

class GeneratedApproach:
    PICK_JOINTS=np.array([0.,.265462,-np.pi,-2.031427,0.,-.758715,np.pi/2],np.float32)
    OFFSET=np.array([.4803523,.0085780],np.float32)
    def __init__(self,action_space,observation_space,primitives):
        self.low=np.asarray(action_space.low,np.float32); self.high=np.asarray(action_space.high,np.float32); self.reset(None,None)
    def reset(self,state,info):
        self.phase='select'; self.target=None; self.rack=None; self.total_parts=None; self.retry=0; self.descents=0; self.placed=[]; self.used_slots=[]; self.slot=None; self.failures={}; self.deferred=set()
    @staticmethod
    def _g(s,n,f): return s.get(s.get_object_from_name(n),f)
    @staticmethod
    def _parts(s): return sorted(n for n in s.get_object_names() if n.startswith('part'))
    def _action(self,s,base=None,joints=None,grip=0.):
        a=np.zeros(11,np.float32)
        a[2]=np.clip(-self._g(s,'robot','pos_base_rot'),-.2,.2)
        if base is not None:
            a[0]=np.clip(base[0]-self._g(s,'robot','pos_base_x'),-.2,.2); a[1]=np.clip(base[1]-self._g(s,'robot','pos_base_y'),-.2,.2)
        if joints is not None:
            for i in range(7): a[3+i]=np.clip(joints[i]-self._g(s,'robot','joint_%d'%(i+1)),-.2,.2)
        a[10]=grip
        return np.clip(a,self.low,self.high).astype(np.float32)
    def _at(self,s,base=None,joints=None,tol=.004):
        if base is not None and (abs(self._g(s,'robot','pos_base_x')-base[0])>tol or abs(self._g(s,'robot','pos_base_y')-base[1])>tol): return False
        if joints is not None:
            for i in range(7):
                if abs(self._g(s,'robot','joint_%d'%(i+1))-joints[i])>tol:return False
        return True
    def _supported(self,s,p):
        return abs(self._g(s,p,'pose_x')-self.rack[0])<.105 and abs(self._g(s,p,'pose_y')-self.rack[1])<.155 and self._g(s,p,'pose_z')<.115
    def _shape_correction(self,s,p):
        obj=s.get_object_from_name(p)
        if obj.type.name=='Kinematic3DTriangle':
            # The two right-triangle meshes occupy opposite halves of their 0.1 m
            # bounding square, so aim toward the interior of the appropriate half.
            if self._g(s,p,'triangle_type')>.5:return np.array([.03,.03],np.float32)
            return np.zeros(2,np.float32)
        return np.zeros(2,np.float32)
    def _retry_offset(self):
        o=[(0,0),(-.01,0),(.01,0),(0,-.01),(0,.01),(-.02,0),(.02,0),(0,-.02),(0,.02),(-.015,-.015),(-.015,.015),(.015,-.015),(.015,.015),(-.03,0),(.03,0),(0,-.03),(0,.03)]
        return np.asarray(o[min(self.retry,16)],np.float32)
    def _choose_slot(self,s,p):
        obj=s.get_object_from_name(p)
        if obj.type.name=='Kinematic3DCuboid':
            cells=[(0.,-.07),(0.,.07),(.04,-.07),(.04,.07),(0.,0),(.04,0)]
        else:
            # Small row offsets are contact-sensitive: +0.05 reaches the rack floor,
            # while +0.07 catches the shallow lip and detaches a few mm too high.
            cells=[(-.03,-.07),(-.03,.05),(.07,-.07),(.07,.05),(-.03,0),(.07,0)]
        free=[c for c in cells if not any(abs(c[0]-u[0])<.07 and abs(c[1]-u[1])<.04 for u in self.used_slots)] or cells
        c=free[0]
        if s.get_object_from_name(p).type.name=='Kinematic3DTriangle' and self._g(s,p,'triangle_type')<.5:
            c=(0.,.05) if self.placed else (0.,-.07)
        return c,np.array([self.rack[0]+c[0],self.rack[1]+c[1]],np.float32)
    def get_action(self,s):
        if self.rack is None:self.rack=np.array([self._g(s,'rack','pose_x'),self._g(s,'rack','pose_y')],np.float32)
        if self.total_parts is None:self.total_parts=len(self._parts(s))
        holding=self._g(s,'robot','grasp_active')>.5
        if holding and self.target is None:
            grasped=[p for p in self._parts(s) if self._g(s,p,'grasp_active')>.5]
            if grasped:
                self.target=grasped[0]
                self.slot_key,self.slot=self._choose_slot(s,self.target)
                self.lift_amount=.30 if self.placed and self.total_parts>=3 else .15
        if holding and self.target is not None and self.phase not in ('lift','carry','descend','release'):self.phase='lift'
        if holding and self.target is None:return self._action(s,grip=1.)
        if self.phase=='select':
            # Placement is not permanent: a later carry can brush an earlier part
            # off the shallow rack.  Reconcile memory with the current observation
            # so displaced parts are selected again instead of being forgotten.
            self.placed=[p for p in self.placed if self._supported(s,p)]
            all_cs=[p for p in self._parts(s) if p not in self.placed and not self._supported(s,p)]
            cs=[p for p in all_cs if p not in self.deferred]
            if not cs and all_cs:self.deferred.clear(); cs=all_cs
            if not cs:return self._action(s,grip=1.)
            # Fill from the negative-y row across the rack. Choosing a part on the
            # same side avoids carrying later objects across already packed ones.
            desired=[-.07,.07,-.07,.07,0.,0.][len(self.placed)%6]
            def rank(p):
                o=s.get_object_from_name(p)
                # Both triangle meshes have one especially reliable negative-y
                # contact target. Reserve it by packing triangles before cuboids.
                if o.type.name=='Kinematic3DTriangle':
                    shape_rank=0 if self._g(s,p,'triangle_type')>.5 else 1
                else: shape_rank=2
                return (shape_rank,abs((self._g(s,p,'pose_y')-self.rack[1])-desired))
            self.target=min(cs,key=rank); self.retry=0; self.phase='approach'; self.slot_key,self.slot=self._choose_slot(s,self.target); self.lift_amount=.30 if self.placed and self.total_parts>=3 else .15
        if self.phase=='approach':
            xy=np.array([self._g(s,self.target,'pose_x'),self._g(s,self.target,'pose_y')],np.float32)
            base=xy-self.OFFSET+self._shape_correction(s,self.target)+self._retry_offset()
            if not self._at(s,base,self.PICK_JOINTS):return self._action(s,base,self.PICK_JOINTS,1.)
            self.phase='close'; return self._action(s,base,self.PICK_JOINTS,-1.)
        if self.phase=='close':
            if holding:self.phase='lift'
            else:
                self.retry=(self.retry+1)%17; self.phase='approach'; return self._action(s,grip=1.)
        if self.phase=='lift':
            lifted=self.PICK_JOINTS.copy(); lifted[1]-=self.lift_amount
            if not self._at(s,joints=lifted):return self._action(s,joints=lifted,grip=0.)
            is_cube=s.get_object_from_name(self.target).type.name=='Kinematic3DCuboid'
            self.phase='carry'; self.carry_stage=(0 if self.placed and is_cube and self.lift_amount<.2 else 2); self.carry_start_y=self._g(s,'robot','pos_base_y')
        if self.phase=='carry':
            lifted=self.PICK_JOINTS.copy(); lifted[1]-=self.lift_amount
            base=self.slot-self.OFFSET-np.array([.0235403,0],np.float32)
            # Route behind the rack before moving laterally. A direct diagonal path
            # can drag a held object through an already packed part and deadlock.
            if self.carry_stage==0:
                waypoint=np.array([-.36,self.carry_start_y],np.float32)
                if not self._at(s,waypoint,lifted):return self._action(s,waypoint,lifted,0.)
                self.carry_stage=1
            if self.carry_stage==1:
                waypoint=np.array([-.36,base[1]],np.float32)
                if not self._at(s,waypoint,lifted):return self._action(s,waypoint,lifted,0.)
                self.carry_stage=2
            if not self._at(s,base,lifted):return self._action(s,base,lifted,0.)
            self.phase='descend'; self.descents=0
        if self.phase=='descend':
            if not holding:
                if self._supported(s,self.target):
                    self.placed.append(self.target); self.used_slots.append(self.slot_key)
                    self.phase='select'
                else:
                    self.failures[self.target]=self.failures.get(self.target,0)+1
                    if self.failures[self.target]>=2:self.deferred.add(self.target)
                    self.phase='retract'
                return self._action(s,grip=1.)
            # Coarse increments deliberately press slightly into the support; the
            # simulator's grasp constraint then releases and the part settles.
            obj=s.get_object_from_name(self.target)
            type_zero=obj.type.name=='Kinematic3DTriangle' and self._g(s,self.target,'triangle_type')<.5
            extra=8 if self.lift_amount>.2 else 0
            seq=[.02]*((7 if type_zero else 10)+extra)
            if self.descents<len(seq):
                a=self._action(s,grip=0.); a[4]=seq[self.descents]; self.descents+=1; return a
            self.phase='release'
        if self.phase=='release':
            if not holding:
                if self._supported(s,self.target):
                    self.placed.append(self.target); self.used_slots.append(self.slot_key); self.phase='select'
                else:
                    self.failures[self.target]=self.failures.get(self.target,0)+1
                    if self.failures[self.target]>=2:self.deferred.add(self.target)
                    self.phase='retract'
                return self._action(s,grip=1.)
            # Opening alone does not detach a free-space grasp. At support contact a
            # tiny wrist-down pulse supplies the release force without displacing it.
            a=self._action(s,grip=1.)
            type_zero=(s.get_object_from_name(self.target).type.name=='Kinematic3DTriangle' and self._g(s,self.target,'triangle_type')<.5)
            if type_zero:
                a[6]=-.02
            elif s.get_object_from_name(self.target).type.name=='Kinematic3DCuboid':
                a[9]=.05
            elif self._g(s,'robot','joint_4')>-2.59:
                a[6]=-.01
            else:
                # If the wrist reaches its lower limit (notably triangle type 0),
                # scrape gently toward the tray lip to break the grasp constraint.
                a[0]=-.01
            return a
        if self.phase=='push':
            if self._supported(s,self.target):
                self.placed.append(self.target); self.used_slots.append(self.slot_key); self.phase='select'; return self._action(s,grip=1.)
            self.push_steps+=1
            if self.push_steps>10:self.phase='select'; return self._action(s,grip=1.)
            bx=self._g(s,'robot','pos_base_x'); by=self._g(s,'robot','pos_base_y')
            return self._action(s,base=np.array([bx,by+.02],np.float32),grip=1.)
        if self.phase=='retract':
            lifted=self.PICK_JOINTS.copy(); lifted[1]-=.15
            if not self._at(s,joints=lifted):return self._action(s,joints=lifted,grip=1.)
            self.phase='select'; return self._action(s,grip=1.)
        return self._action(s,grip=1.)
