# heuristics/graph.py
#
# Heuristics for graph map.
#
# @author: mike
# @created: 2020-07-22
#

import math

def piglet_heuristic(domain, current_state, goal_state):
    return straight_heuristic(current_state, goal_state)

# In graph map this heuristic may not admissible if the distance derived by give coordinate is lager than given edge weight.
def straight_heuristic(current_state, goal_state):
    return round(math.sqrt((current_state.get_location()[0] - goal_state.get_location()[0])**2 + (current_state.get_location()[1] - goal_state.get_location()[1])**2), 5)