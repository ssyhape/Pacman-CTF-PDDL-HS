import os, sys
from typing import List, Tuple

from lib_piglet.domains.pddl import pddl, pddl_state
from lib_piglet.expanders.pddl_expander import pddl_expander
from lib_piglet.search.graph_search import graph_search
from lib_piglet.search.search_node import compare_node_g, compare_node_f
from lib_piglet.heuristics.pddl_h import piglet_heuristic
from lib_piglet.utils.data_structure import queue,stack,bin_heap
from lib_piglet.cli.cli_tool import statistic_template, print_header
from lib_piglet.utils.pddl_parser import PDDL_Parser, Action



class pddl_solver:
    
    def __init__(self, domain_file: str, heuristic_function = piglet_heuristic):
        self.domain_file_ : str = domain_file 
        self.domain_ : pddl = pddl(domain_file)
        self.parser_ : PDDL_Parser = self.domain_.parser_
        self.expander_ = pddl_expander(self.domain_)
        self.engine_ : graph_search = graph_search(bin_heap(compare_node_f), self.expander_, heuristic_function=piglet_heuristic)
    
    def read_problem(self, problem_file:str):
        self.domain_.read_problem(problem_file)

    def get_parser(self):
        return self.parser_
    
    def solve(self,time_limit: int = None) -> List[Tuple[Action,pddl_state]]:
        if time_limit is not None:
            self.engine_.time_limit_ = time_limit
        else:
            self.engine_.time_limit_ = sys.maxsize
            
        self.domain_.set_start_goal()
        solution = self.engine_.get_path(self.domain_.start_state_,self.domain_.goal_state_)

        plan = []
        if solution is None:
            return plan
        for n in solution.paths_:
            if n.state_.from_action_ is None:
                continue
            plan.append((n.state_.from_action_,n.state_))
        
        return plan
    
    def satisfyPrecondition(self, state: list, action: Action):
        return self.domain_.applicable(frozenset(state), action.positive_preconditions, action.negative_preconditions)
    
    def matchEffect(self, state: list, action: Action):
        state_set = frozenset(state)
        return action.add_effects.issubset(state_set) and action.del_effects.isdisjoint(state_set)
        

        
        

