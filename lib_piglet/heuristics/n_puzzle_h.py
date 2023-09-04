# heuristics/n_puzzle_h.py
#
# Heuristics for n_puzzle problem.
#
# @author: mike
# @created: 2020-07-22
#

import math, copy, sys,json
from lib_piglet.domains.n_puzzle import puzzle_state, n_puzzle
from lib_piglet.expanders import n_puzzle_expander
from lib_piglet.search import dijkstra_search
from lib_piglet.utils.data_structure import bin_heap
from lib_piglet.search.search_node import compare_node_g

pattern_database = {}
pattern_database_pattern_width = 0

# piglet cli will use this function as heuristic.
def piglet_heuristic(domain,current_state, goal_state):
    return sum_manhattan_heuristic(current_state, goal_state)

def num_wrong_heuristic(current_state: puzzle_state, goal_state: puzzle_state):
    length = len(goal_state.state_list_)
    width = math.sqrt(length)
    h = 0
    for g in range(0,length):
        if goal_state.state_list_[g] != current_state.state_list_[g]:
            h+=1
    return h

def sum_manhattan_heuristic(current_state: puzzle_state, goal_state: puzzle_state):
    length = len(goal_state.state_list_)
    width = math.sqrt(length)
    h = 0
    for g in range(0,length):
        if goal_state.state_list_[g] == "x":
            continue
        c = current_state.state_list_.index(goal_state.state_list_[g])
        h += abs(c//width - g//width) + abs(c%width - g%width)
    return h


def sum_straight_heuristic(current_state, goal_state):
    length = len(goal_state.state_list_)
    width = math.sqrt(length)
    h = 0
    for g in range(0, length):
        if goal_state.state_list_[g] == "x":
            continue
        c = current_state.state_list_.index(goal_state.state_list_[g])
        h += round(math.sqrt((c // width - g // width)**2 + (c % width - g % width)**2))
    return h

############
# Codes related to pattern database
############
def extract_fringe_pattern(state: puzzle_state):
    pattern_state = copy.deepcopy(state)
    width = int(math.sqrt(len(pattern_state.state_list_)))
    for i in range(0,len(pattern_state.state_list_)):
        if pattern_state.state_list_[i]!="x" and pattern_state.state_list_[i]//width != 0 and pattern_state.state_list_[i]%width != 0:
            pattern_state.state_list_[i] = 0
    return pattern_state

def extract_corner_pattern(state: puzzle_state):
    pattern_state = copy.deepcopy(state)
    size = len(pattern_state.state_list_)
    for i in range(0,len(pattern_state.state_list_)):
        if pattern_state.state_list_[i]!="x" and pattern_state.state_list_[i]>size//2:
            pattern_state.state_list_[i] = 0
    return pattern_state

def run_dijkstra_n_puzzle(target_state: puzzle_state):
    puzzle_width = int(math.sqrt(len(target_state.state_list_)))
    puzzle = n_puzzle(puzzle_width)
    expander = n_puzzle_expander.n_puzzle_expander(puzzle)
    open_list = bin_heap(compare_node_g)
    search_engine = dijkstra_search.dijkstra_search(open_list,expander)
    return search_engine.get_path(target_state)

def build_corner_pattern_database(goal_state: puzzle_state):
    goal_pattern = extract_corner_pattern(goal_state)
    return run_dijkstra_n_puzzle(goal_pattern).paths_

def build_fringe_pattern_database(goal_state: puzzle_state):
    goal_pattern = extract_fringe_pattern(goal_state)
    return run_dijkstra_n_puzzle(goal_pattern).paths_

def corner_pattern_database_heuristic(current_state: puzzle_state,goal_state: puzzle_state):
    global pattern_database,pattern_database_pattern_width
    current_pattern = extract_corner_pattern(current_state)
    if len(pattern_database) == 0 or len(goal_state.state_list_)!= pattern_database_pattern_width:
        print("Building pattern database ... ...")
        pattern_database = build_corner_pattern_database(goal_state)
        pattern_database_pattern_width = len(goal_state.state_list_)
        sys.stdout.write("\033[F")
    return pattern_database[current_pattern]

def fringe_pattern_database_heuristic(current_state: puzzle_state,goal_state: puzzle_state):
    global pattern_database,pattern_database_pattern_width
    current_pattern = extract_fringe_pattern(current_state)
    if len(pattern_database) == 0 or len(goal_state.state_list_)!= pattern_database_pattern_width:
        print("Building pattern database ... ...")
        pattern_database = build_fringe_pattern_database(goal_state)
        pattern_database_pattern_width = len(goal_state.state_list_)
        sys.stdout.write("\033[F")
    return pattern_database[current_pattern]






