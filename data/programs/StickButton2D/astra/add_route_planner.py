p='approach_planned.py';s=open(p).read()
s=s.replace('        self.held_geometry=None\n','        self.held_geometry=None\n        self.pick_first=self.choose_pick_first(state)\n',1)
pos=s.index('    def read(')
s=s[:pos]+'''    def choose_pick_first(self,s):
        robot=next(iter(s.get_objects(self.space.get_type('crv_robot'))))
        r=self.read(s,robot,['x','y','theta','arm_joint'])
        stick=next(iter(s.get_objects(self.space.get_type('rectangle'))))
        st=self.read(s,stick,['x','y','theta','width','height'])
        buttons=[self.read(s,b,['x','y']) for b in s.get_objects(self.space.get_type('circle')) if b.name.startswith('button')]
        if not any(b[1]>1.34 for b in buttons):return False
        if len(buttons)>12:return True
        def steps(r,g):
            return max(np.max(abs(g[:2]-r[:2]))/.05,abs(wrap(g[2]-r[2]))/.196)
        def estimate(r,buttons,direct):
            r=r.copy();buttons=list(buttons);cost=0.
            if direct:
                while True:
                    goals=[(self.direct_goal(r,b),k) for k,b in enumerate(buttons) if b[1]<=1.34]
                    goals=[(steps(r,g),g,k) for g,k in goals if g is not None]
                    if not goals:break
                    c,g,k=min(goals,key=lambda a:a[0]);cost+=c;r=g;buttons.pop(k)
            sx,sy,_,w,h=st
            picks=[(max(abs(px-r[0])/.05,abs(wrap(t-r[2]))/.196),t,px) for t,px in [(math.pi,sx+w+.218),(0.,sx-.218)] if .101<px<3.399]
            c,t,px=min(picks)
            stage_y=max(.25,min(r[1],sy-.25))
            cost+=abs(r[1]-stage_y)/.05+c+abs(sy-.025-stage_y)/.05+2
            r=np.array([px,sy-.025,t,.2]);tool=st.copy()
            while buttons:
                goals=[(self.held_goal(r,tool,b),k) for k,b in enumerate(buttons)]
                goals=[(steps(r,g),g,k) for g,k in goals if g is not None]
                if not goals:return 1e5
                c,g,k=min(goals,key=lambda a:a[0]);cost+=c
                dt=wrap(g[2]-r[2]);co,si=math.cos(dt),math.sin(dt)
                tool[:2]=g[:2]+np.array([[co,-si],[si,co]])@(tool[:2]-r[:2]);tool[2]+=dt
                r=g;buttons.pop(k)
            return cost
        return estimate(r,buttons,False)<estimate(r,buttons,True)
''' +s[pos:]
s=s.replace('            if candidates:\n','            if candidates and not self.pick_first:\n')
open(p,'w').write(s)
p='test_planned.py';s=open('test_approach.py').read().replace('from approach import GeneratedApproach','from approach_planned import GeneratedApproach');open(p,'w').write(s)
