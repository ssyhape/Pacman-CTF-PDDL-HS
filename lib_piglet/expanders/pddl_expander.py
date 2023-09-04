
from lib_piglet.expanders.base_expander import base_expander
from lib_piglet.domains.pddl import pddl, pddl_state
from lib_piglet.search.search_node import search_node
from lib_piglet.utils.pddl_parser import Action
from enum import IntEnum



class pddl_action:

    def __init__(self, move: Action, cost:int):
        self.move_: Action = move
        self.cost_: int = cost
        
class pddl_expander(base_expander):

    def __init__(self, domain:pddl):
        self.domain_: pddl = domain
        self.succ_: list = []
    
    def expand(self, current:search_node):
        self.succ_.clear()
        current_state:pddl_state = current.state_
        for act in self.domain_.ground_actions_:
            if self.domain_.applicable(current_state.state_set_, act.positive_preconditions, act.negative_preconditions):
                new_state = self.domain_.apply(current_state.state_set_, act.add_effects, act.del_effects)
                
                self.succ_.append((pddl_state(new_state, act), pddl_action(act,1)))

        return self.succ_[:]
    
    def __str__(self):
        return self.domain_.problem_path_

class pddl_greedy_relaxation_expander(base_expander):

    def __init__(self, domain:pddl):
        self.domain_: pddl = domain
        self.succ_: list = []
        self.empty_set_: frozenset = frozenset()
    
    def expand(self, current:search_node):
        self.succ_.clear()
        current_state:pddl_state = current.state_
        for act in self.domain_.ground_actions_:
            if self.domain_.applicable(current_state.state_set_, act.positive_preconditions, self.empty_set_):
                new_state = self.domain_.apply(current_state.state_set_, act.add_effects, self.empty_set_)
                if new_state != current_state.state_set_:
                    self.succ_.append((pddl_state(new_state, act), pddl_action(act,1)))
                    break
        return self.succ_[:]
    
    def __str__(self):
        return self.domain_.problem_path_

class pddl_optimal_relaxation_expander(base_expander):

    def __init__(self, domain:pddl):
        self.domain_: pddl = domain
        self.succ_: list = []
        self.empty_set_: frozenset = frozenset()
    
    def expand(self, current:search_node):
        self.succ_.clear()
        current_state:pddl_state = current.state_
        for act in self.domain_.ground_actions_:
            if self.domain_.applicable(current_state.state_set_, act.positive_preconditions, self.empty_set_):
                new_state = self.domain_.apply(current_state.state_set_, act.add_effects, self.empty_set_)
                if new_state != current_state.state_set_:
                    self.succ_.append((pddl_state(new_state, act), pddl_action(act,1)))
        return self.succ_[:]
    
    def __str__(self):
        return self.domain_.problem_path_




    


