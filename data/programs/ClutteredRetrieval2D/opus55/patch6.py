src = open('approach.py').read()
for fn in ['plan_grasp', 'plan_place', 'plan_dump']:
    import re
    m = re.search(r'    def %s\(self[^\n]*\n' % fn, src)
    src = src[:m.end()] + '        time_budget *= self.tscale()\n' + src[m.end():]
src = src.replace('''    def elapsed(self):''', '''    def tscale(self):
        e = self.elapsed()
        if e < 25:
            return 1.0
        if e < 40:
            return 0.5
        return 0.2

    def elapsed(self):''')
open('approach.py','w').write(src)
