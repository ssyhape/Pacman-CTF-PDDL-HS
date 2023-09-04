# heuristics/gridmap_h.py
#
# Heuristics for gridmap.
#
# @author: mike
# @created: 2020-07-22
#

import math

def piglet_heuristic(domain,current_state, goal_state):
    return manhattan_heuristic(current_state, goal_state)

def pigelet_multi_agent_heuristic(domain,current_state, goal_state):
    h = 0
    for agent, loc in current_state.agent_locations_.items():
        h += manhattan_heuristic(loc, goal_state.agent_locations_[agent])
    return h

def manhattan_heuristic(current_state, goal_state):
    return NotImplementedError

def straight_heuristic(current_state, goal_state):
    return NotImplementedError

def octile_heuristic(current_state, goal_state):
    return NotImplementedError

def differential_heuristic(domain, current_state, goal_state):
    return NotImplementedError