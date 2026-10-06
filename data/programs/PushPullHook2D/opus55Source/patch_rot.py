s=open('approach.py').read()
s=s.replace("""            cost = dist[bi, bj] + pcost
            if best""","""            nr = math.ceil(abs(wrap(bpose[2] - th0)) / MAX_DTH - 1e-9)
            cost = max(dist[bi, bj] + 1, nr) + pcost
            if best""")
s=s.replace("""                cost = dist[bi, bj] + len(carry) + 1
""","""                nr = math.ceil(abs(wrap(bpose[2] - th0)) / MAX_DTH - 1e-9)
                cost = max(dist[bi, bj] + 1, nr) + len(carry) + 1
""")
open('approach.py','w').write(s)
