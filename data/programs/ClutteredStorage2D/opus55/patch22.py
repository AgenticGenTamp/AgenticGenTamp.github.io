src=open('approach.py').read()
old="""        if self.phase == 'start':
            self.phase = 'nav'
            self.path = None
            cx = b['center'][0]"""
new="""        if self.phase == 'start':
            self.failed[name] = self.failed.get(name, 0) + 0.5
            self.phase = 'nav'
            self.path = None
            cx = b['center'][0]"""
assert src.count(old)==1
open('approach.py','w').write(src.replace(old,new))
