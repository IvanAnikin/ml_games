

def show_env_props(env):

    env_name = env.unwrapped.spec.id
    num_states = env.observation_space.shape
    num_actions = env.action_space.n


    print("Env Name ->  {}".format(env_name))

    print("Size of State Space ->  {}".format(num_states))
    print("Size of Action Space ->  {}".format(num_actions))