import sys
import planner, stages
V = "-v" in sys.argv
for kv in [a for a in sys.argv[2:] if a != "-v"]:
    k, v = kv.split('=')
    mod, name = k.split('.')
    import approach; setattr({'planner': planner, 'stages': stages, 'approach': approach}[mod], name, eval(v))
sys.argv = [sys.argv[0], sys.argv[1]] + (["-v"] if V else [])
exec(open('test_approach.py').read())
