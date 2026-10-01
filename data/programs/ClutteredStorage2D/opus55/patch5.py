src=open('test_approach.py').read()
src=src.replace("""    seeds = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(10)
    oc = int(sys.argv[3]) if len(sys.argv) > 3 else None""","""    if sys.argv[1] == 'list':
        seeds = [int(x) for x in sys.argv[2].split(',')]
    else:
        seeds = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(10)
    oc = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '0' else None""")
open('test_approach.py','w').write(src)
