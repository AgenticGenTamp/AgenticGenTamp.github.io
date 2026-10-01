src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""        outs.sort(key=lambda nb: self.failed.get(nb[0], 0) * 10 +
                  math.hypot(nb[1]['center'][0] - self.rx, nb[1]['center'][1] - self.ry))
        return ('fetch', outs[0][0])""","""        def fcost(nb):
            opts = self._grasp_options(nb[0])
            base = opts[0][0] if opts else 1e4
            return self.failed.get(nb[0], 0) * 200 + base
        outs.sort(key=fcost)
        return ('fetch', outs[0][0])""")
open('approach.py','w').write(src)
