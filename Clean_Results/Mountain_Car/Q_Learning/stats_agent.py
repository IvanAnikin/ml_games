


import Clean_Results.Agents.storage_agent as storage_agent


def best_rewards_params(score_count_start_position = 4000, LEARNING_RATES = [0.05, 0.10, 0.15, 0.20], EPSILONS = [0.5], END_EPSILON_DECAYING_POSITIONS = [2.0], DISCOUNTS = [0.95], DISCRETE_OS_SIZES = [20], show_every = 4000, episodes = 5000, stats_every = 100):

    total_score = 0
    max_score = -10000

    score_count_start = score_count_start_position/stats_every

    best_game_name = ""

    for learning_rate_cycle in range(len(LEARNING_RATES)):
        for epsilon_cycle in range(len(EPSILONS)):
            for end_epsilon_decaying_cycle in range(len(END_EPSILON_DECAYING_POSITIONS)):
                for discount_cycle in range(len(DISCOUNTS)):
                    for discrete_os_size_cycle in range(len(DISCRETE_OS_SIZES)):
                        total_score = 0

                        learning_rate = LEARNING_RATES[learning_rate_cycle]
                        epsilon = EPSILONS[epsilon_cycle]
                        end_epsilon_decaying = END_EPSILON_DECAYING_POSITIONS[end_epsilon_decaying_cycle]
                        discount = DISCOUNTS[discount_cycle]
                        discrete_os_size = [DISCRETE_OS_SIZES[discrete_os_size_cycle],
                                            DISCRETE_OS_SIZES[discrete_os_size_cycle]]

                        NAME = "ep-{}__stats-{}__lr-{}__eps-{}__epsDec-{}__disc-{}__size-{}".format(episodes,
                                                                                                    stats_every,
                                                                                                    learning_rate,
                                                                                                    epsilon,
                                                                                                    end_epsilon_decaying,
                                                                                                    discount,
                                                                                                    discrete_os_size)

                        data_avg = storage_agent.load_np(NAME)

                        for data in range(len(data_avg)):
                            if data >= score_count_start:
                                total_score += data_avg[data]

                        if total_score > max_score:
                            best_game_name = NAME
                            max_score = total_score

    return best_game_name
