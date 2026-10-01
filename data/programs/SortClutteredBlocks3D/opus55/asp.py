from env_client import make_env
env=make_env(); s=env.action_space; print(vars(s) if hasattr(s,'__dict__') else s)
