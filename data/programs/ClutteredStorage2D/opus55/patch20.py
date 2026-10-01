src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""        outs.sort(key=fcost)
        return ('fetch', outs[0][0])""","""        def rough(nb):
            c = nb[1]['center']
            sgx = min(max(c[0], self.sx1 + 0.15), self.sx1 + self.sw1 - 0.15)
            return self.failed.get(nb[0], 0) * 10 + max(abs(c[0] - self.rx), abs(c[1] - self.ry)) \\
                + max(abs(c[0] - sgx), abs(c[1] - 2.3))
        outs.sort(key=rough)
        outs = outs[:4]
        outs.sort(key=fcost)
        return ('fetch', outs[0][0])""")
open('approach.py','w').write(src)
