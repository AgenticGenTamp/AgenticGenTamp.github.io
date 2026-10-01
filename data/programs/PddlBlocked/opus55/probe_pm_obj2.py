from env_client import make_env
env=make_env(); obs,_=env.reset(seed=2)
T={t.name:t for t in env.observation_space.type_features}
for tn in ['surface','block']:
    for o in obs.get_objects(T[tn]):
        print(o.name, obs.get_object_feature_names(o) if hasattr(obs,'get_object_feature_names') else '', [round(float(x),3) for x in obs[o]] if hasattr(obs,'__getitem__') else '')
