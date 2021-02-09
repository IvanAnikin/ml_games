
from collections import OrderedDict

# Game
WORLD = (1, 1)

# Distance to castle for every level
WIN_DISTANCES = OrderedDict([((1, 1), 3266), ((1, 2), 3266), ((1, 3), 2514), ((1, 4), 2430),
                             ((2, 1), 3298), ((2, 2), 3266), ((2, 3), 3682), ((2, 4), 2430),
                             ((3, 1), 3298), ((3, 2), 3442), ((3, 3), 2498), ((3, 4), 2430),
                             ((4, 1), 3698), ((4, 2), 3266), ((4, 3), 2434), ((4, 4), 2942),
                             ((5, 1), 3282), ((5, 2), 3298), ((5, 3), 2514), ((5, 4), 2429),
                             ((6, 1), 3106), ((6, 2), 3554), ((6, 3), 2754), ((6, 4), 2429),
                             ((7, 1), 2962), ((7, 2), 3266), ((7, 3), 3682), ((7, 4), 3453),
                             ((8, 1), 6114), ((8, 2), 3554), ((8, 3), 3554), ((8, 4), 4989)])

# The flagpole is 40 meters before the castle
LEVEL_WIN_DIST = WIN_DISTANCES[WORLD] - 40

# Penalty for dying in reward function
DEATH_PENALTY = 100