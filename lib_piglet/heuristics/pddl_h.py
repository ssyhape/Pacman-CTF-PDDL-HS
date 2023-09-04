import math, copy, sys
from lib_piglet.expanders.pddl_expander import pddl_optimal_relaxation_expander,pddl_greedy_relaxation_expander
from lib_piglet.domains.pddl import pddl_goal
from lib_piglet.search.graph_search import graph_search
from lib_piglet.utils.data_structure import bin_heap
from lib_piglet.search.search_node import compare_node_g, compare_node_f


def piglet_heuristic(domain,current_state, goal_state):
    return optimal_delete_relaxation_h(domain, current_state,goal_state)

def greedy_delete_relaxation_h(domain, current_state,goal_state):
    if current_state in domain.h_dict:
        return domain.h_dict[current_state]

    relaxed_expander = pddl_greedy_relaxation_expander(domain)
    relaxed_search = graph_search(bin_heap(compare_node_g), relaxed_expander)
    relaxed_goal = pddl_goal(copy.deepcopy(goal_state.goal_pos_), frozenset())
    relaxed_solution = relaxed_search.get_path(copy.deepcopy(current_state), relaxed_goal)
    if relaxed_solution is None:
        domain.h_dict[current_state] = sys.maxsize
        return sys.maxsize
    else:
        domain.h_dict[current_state] = relaxed_solution.cost_
        return relaxed_solution.cost_

def optimal_delete_relaxation_h(domain, current_state,goal_state):
    if current_state in domain.h_dict:
        return domain.h_dict[current_state]

    relaxed_expander = pddl_optimal_relaxation_expander(domain)
    relaxed_search = graph_search(bin_heap(compare_node_g), relaxed_expander)
    relaxed_goal = pddl_goal(copy.deepcopy(goal_state.goal_pos_), frozenset())
    relaxed_solution = relaxed_search.get_path(copy.deepcopy(current_state), relaxed_goal)
    if relaxed_solution is None:
        domain.h_dict[current_state] = sys.maxsize
        return sys.maxsize
    else:
        domain.h_dict[current_state] = relaxed_solution.cost_
        return relaxed_solution.cost_






