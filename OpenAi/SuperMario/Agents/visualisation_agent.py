

def show_env_props(env):
    env_name = env.unwrapped.spec.id
    num_states = env.observation_space.shape#[0]
    num_actions = env.action_space.n

    #upper_bound = env.action_space.high[0]
    #lower_bound = env.action_space.low[0]


    print("Env Name ->  {}".format(env_name))

    print("Size of State Space ->  {}".format(num_states))
    print("Size of Action Space ->  {}".format(num_actions))

    #print("Max Value of Action ->  {}".format(upper_bound))
    #print("Min Value of Action ->  {}".format(lower_bound))